from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np
from torch.utils.data import Dataset

sys.path.append('..')
from dataloaders.msrvtt_protocol import load_trusted_manifest
from dataloaders.rawframes_util import RawFramesExtractor
from dataloaders.rawvideo_util import RawVideoExtractor

# Late-imported to avoid dragging torch dependencies into modules that only
# consume the manifest utilities.
try:
    from dataloaders.crf_pairing import CRFBatchIndex, RANK_TO_CRF
except ImportError:  # pragma: no cover — Task 4 not yet installed
    CRFBatchIndex = None  # type: ignore
    RANK_TO_CRF = {1: 30, 2: 38, 3: 46}


def _validate_tqfs_dependency(slice_framepos):
    if slice_framepos == 3 and importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "slice_framepos=3 requires scikit-learn for deterministic TQFS; "
            "install the pinned project requirements instead of silently falling back"
        )


def _split_attr_into_blocks(text: str, num_blocks: int = 4):
    """Split Qwen-generated attributes into semantic blocks.

    Supports two prompt formats:
      v1: 【ENTITIES】, 【ACTIONS】, 【APPEARANCE & DETAILS】, 【SCENE】, 【TEXT/OCR】
      v2: SUBJECTS:, ACTIONS:, OBJECTS:, SETTING:

    Returns a list of block strings (length==num_blocks). Missing blocks padded with "".
    """
    num_blocks = int(num_blocks) if num_blocks is not None else 4
    if num_blocks <= 0:
        num_blocks = 1
    if not isinstance(text, str):
        return [""] * num_blocks
    s = text.strip()
    if not s:
        return [""] * num_blocks

    # --- Try v2 format first (SUBJECTS:/ACTIONS:/OBJECTS:/SETTING:) ---
    v2_headings = ["SUBJECTS:", "ACTIONS:", "OBJECTS:", "SETTING:"]
    v2_pos = []
    for h in v2_headings:
        i = s.find(h)
        if i >= 0:
            v2_pos.append((i, h))

    if len(v2_pos) >= 2:
        v2_pos.sort(key=lambda x: x[0])
        chunks = {}
        for idx, (start, h) in enumerate(v2_pos):
            content_start = start + len(h)
            end = v2_pos[idx + 1][0] if idx + 1 < len(v2_pos) else len(s)
            chunks[h] = s[content_start:end].strip()

        blocks = [
            chunks.get("SUBJECTS:", ""),
            chunks.get("ACTIONS:", ""),
            chunks.get("OBJECTS:", ""),
            chunks.get("SETTING:", ""),
        ]
        blocks = [(b if isinstance(b, str) else "") for b in blocks]
        if len(blocks) < num_blocks:
            blocks = blocks + [""] * (num_blocks - len(blocks))
        return blocks[:num_blocks]

    # --- Fallback: v1 format (【ENTITIES】/【ACTIONS】/...) ---
    v1_headings = [
        "【ENTITIES】",
        "【ACTIONS】",
        "【APPEARANCE & DETAILS】",
        "【SCENE】",
        "【TEXT/OCR】",
    ]
    pos = []
    for h in v1_headings:
        i = s.find(h)
        if i >= 0:
            pos.append((i, h))
    if not pos:
        blocks = [s] + [""] * (num_blocks - 1)
        return blocks[:num_blocks]
    pos.sort(key=lambda x: x[0])

    chunks = {}
    for idx, (start, h) in enumerate(pos):
        end = pos[idx + 1][0] if idx + 1 < len(pos) else len(s)
        chunks[h] = s[start:end].strip()

    ent = chunks.get("【ENTITIES】", "")
    act = chunks.get("【ACTIONS】", "")
    app = chunks.get("【APPEARANCE & DETAILS】", "")
    scene = chunks.get("【SCENE】", "")
    ocr = chunks.get("【TEXT/OCR】", "")

    if int(num_blocks) == 4:
        if app:
            ent = (ent + "\n" + app).strip() if ent else app
        blocks = [ent, act, scene, ocr]
    else:
        blocks = [ent, act, app, scene, ocr]

    blocks = [(b if isinstance(b, str) else "") for b in blocks]
    if len(blocks) < num_blocks:
        blocks = blocks + [""] * (num_blocks - len(blocks))
    return blocks[:num_blocks]


def _load_attributes_map(path: str):
    """Load attributes mapping.
    Supports:
      - JSON: {video_id: "attributes text", ...}
      - JSONL: each line has {"video_id": "...", "attributes": "..."} (or "attribute"/"attr"/"text")
    """
    if path is None:
        return {}
    path = str(path).strip()
    if not path:
        return {}
    # Support multiple attribute files joined by comma:
    #   --msrvtt_attributes_path "train9k.json,test1k.json"
    # Later files override earlier ones on key collision.
    if "," in path:
        merged = {}
        parts = [p.strip() for p in path.split(",") if p.strip()]
        for p in parts:
            if not p:
                continue
            m = _load_attributes_map(p)
            if isinstance(m, dict) and m:
                merged.update(m)
        print(f"[MSRVTT] Loaded attributes map from {len(parts)} files, merged size={len(merged)}")
        return merged

    if not os.path.exists(path):
        print(f"[MSRVTT] attributes_path not found: {path}")
        return {}

    if path.endswith(".jsonl"):
        m = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                vid = obj.get("video_id")
                if not isinstance(vid, str):
                    continue
                txt = (
                    obj.get("attributes")
                    or obj.get("attribute")
                    or obj.get("attr")
                    or obj.get("text")
                    or ""
                )
                if isinstance(txt, str) and txt.strip():
                    m[vid] = txt
        print(f"[MSRVTT] Loaded attributes map from jsonl: {path}, size={len(m)}")
        return m

    # default: json
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        # keep only string values
        m = {k: v for k, v in obj.items() if isinstance(k, str) and isinstance(v, str) and v.strip()}
        print(f"[MSRVTT] Loaded attributes map from json: {path}, size={len(m)}")
        return m
    print(f"[MSRVTT] Unexpected attributes file schema: {path}")
    return {}



def _read_msrvtt_csv(csv_path: str, need_sentence: bool):
    """Read MSRVTT csv without pandas.

    NOTE: pandas 2.2.3 may crash with TypeError on some environments when reading MSRVTT csv.
    """
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MSRVTT csv_path not found: {csv_path}")

    video_ids = []
    sentences = [] if need_sentence else None
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"MSRVTT csv header not found: {csv_path}")
        if "video_id" not in reader.fieldnames:
            raise ValueError(f"MSRVTT csv missing 'video_id' column: {csv_path}, columns={reader.fieldnames}")
        if need_sentence and "sentence" not in reader.fieldnames:
            raise ValueError(f"MSRVTT csv missing 'sentence' column: {csv_path}, columns={reader.fieldnames}")

        for row in reader:
            vid = row.get("video_id", "")
            if not isinstance(vid, str):
                vid = str(vid)
            vid = vid.strip()
            if not vid:
                continue
            video_ids.append(vid)
            if need_sentence:
                sent = row.get("sentence", "")
                if not isinstance(sent, str):
                    sent = str(sent)
                sentences.append(sent)

    if need_sentence:
        assert sentences is not None
        if len(video_ids) != len(sentences):
            raise ValueError(f"MSRVTT csv length mismatch: video_ids={len(video_ids)} sentences={len(sentences)}")
        return video_ids, sentences
    return video_ids, None


class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            max_words_attrs=None,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            use_attributes=False,
            attributes_path="",
            attr_num_blocks=4,
            multi_sentence_per_video=False,
            expected_captions_per_video=None,
    ):
        self.video_ids, self.sentences = _read_msrvtt_csv(csv_path, need_sentence=True)
        self.multi_sentence_per_video = bool(multi_sentence_per_video)
        self.expected_captions_per_video = expected_captions_per_video
        if self.multi_sentence_per_video:
            counts = []
            seen = set()
            current = None
            current_count = 0
            for video_id in self.video_ids:
                if video_id != current:
                    if video_id in seen:
                        raise ValueError(
                            f"val rows must be contiguous by video_id: {video_id}"
                        )
                    if current is not None:
                        counts.append(current_count)
                    seen.add(video_id)
                    current = video_id
                    current_count = 1
                else:
                    current_count += 1
            if current is not None:
                counts.append(current_count)
            if expected_captions_per_video is not None and any(
                count != expected_captions_per_video for count in counts
            ):
                raise ValueError(
                    f"each val video must have {expected_captions_per_video} captions; "
                    f"observed counts={sorted(set(counts))}"
                )
            self.cut_off_points = np.cumsum(counts).tolist()
            self.sentence_num = len(self.video_ids)
            self.video_num = len(counts)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_words_attrs = max_words if max_words_attrs is None else int(max_words_attrs)
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        self.strategy = 1
        # print('Using uniform sampling without random offset for validation.')
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3]   # 3: TQFS 帧质量采样
        _validate_tqfs_dependency(self.slice_framepos)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=True, strategy=self.strategy)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.use_attributes = bool(use_attributes)
        self.attributes_path = attributes_path
        self.attributes_map = _load_attributes_map(attributes_path) if self.use_attributes else {}
        self.attr_num_blocks = int(attr_num_blocks) if attr_num_blocks is not None else 4

    def __len__(self):
        return len(self.video_ids)

    def _get_text(self, video_id, sentence, max_words=None):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        mw = self.max_words if max_words is None else int(max_words)
        pairs_text = np.zeros((k, mw), dtype=np.int64)
        pairs_mask = np.zeros((k, mw), dtype=np.int64)
        pairs_segment = np.zeros((k, mw), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = mw - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < mw:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == mw
            assert len(input_mask) == mw
            assert len(segment_ids) == mw

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_text_list(self, video_id, sentences, max_words=None):
        """Tokenize a list of sentences into (k, mw) arrays."""
        if not isinstance(sentences, (list, tuple)):
            sentences = [sentences]
        sentences = list(sentences)
        if len(sentences) == 0:
            sentences = [""]

        choice_video_ids = [video_id]
        k = len(sentences)
        mw = self.max_words if max_words is None else int(max_words)
        pairs_text = np.zeros((k, mw), dtype=np.int64)
        pairs_mask = np.zeros((k, mw), dtype=np.int64)
        pairs_segment = np.zeros((k, mw), dtype=np.int64)

        for i, sent in enumerate(sentences):
            if not isinstance(sent, str):
                sent = str(sent)
            words = self.tokenizer.tokenize(sent)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = mw - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < mw:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            if self.slice_framepos == 3:
                raw_video_data = self.rawVideoExtractor.get_tqfs_video_data(
                    video_path, self.max_frames
                )
            else:
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask


    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float32)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}".format(video_id))  # folder

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:    # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:   # extract uniformly
                        sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_index, ...]
                else:
                    video_slice = raw_video_slice
                # 帧序：顺序，逆序，随机打乱
                video_slice = self.rawFramesExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        sentence = self.sentences[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence, max_words=self.max_words)

        if self.use_attributes:
            attrs = self.attributes_map.get(video_id, "")
            if not isinstance(attrs, str) or not attrs.strip():
                attrs = sentence  # fallback
            attr_blocks = _split_attr_into_blocks(attrs, num_blocks=self.attr_num_blocks)
            pairs_text_a, pairs_mask_a, pairs_segment_a, _ = self._get_text_list(
                video_id, attr_blocks, max_words=self.max_words_attrs
            )
        else:
            pairs_text_a = pairs_mask_a = pairs_segment_a = None

        video, video_mask = self._get_rawvideo(choice_video_ids)
        # video, video_mask = self._get_rawframes(choice_video_ids)
        if self.use_attributes:
            return pairs_text, pairs_mask, pairs_segment, pairs_text_a, pairs_mask_a, pairs_segment_a, video, video_mask
        return pairs_text, pairs_mask, pairs_segment, video, video_mask


class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            max_words_attrs=None,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            strategy=1,
            use_attributes=False,
            attributes_path="",
            attr_num_blocks=4,
            split_manifest_path=None,
            expected_captions_per_video=20,
            crf_ladder_root=None,
    ):
        # `crf_ladder_root` opts the dataset into CRF-mode `__getitem__` when the
        # batch index is a `CRFBatchIndex`. Non-CRF training passes remain
        # entirely unchanged.
        self.crf_ladder_root = crf_ladder_root
        self.unfold_sentences = bool(unfold_sentences)
        if not self.unfold_sentences:
            raise ValueError(
                "trusted-v1 MSRVTT training requires unfold_sentences=True"
            )
        if not split_manifest_path:
            raise ValueError(
                "trusted-v1 MSRVTT training requires split_manifest_path"
            )
        if (
            isinstance(expected_captions_per_video, bool)
            or not isinstance(expected_captions_per_video, int)
            or expected_captions_per_video <= 0
        ):
            raise ValueError(
                "expected_captions_per_video must be a positive integer, "
                f"got {expected_captions_per_video!r}"
            )

        self.csv_video_ids, _ = _read_msrvtt_csv(csv_path, need_sentence=False)
        manifest = load_trusted_manifest(split_manifest_path)
        manifest_train_ids = manifest["train_video_ids"]
        self.video_group_ids = {
            video_id: index for index, video_id in enumerate(manifest_train_ids)
        }
        csv_counts = Counter(self.csv_video_ids)
        duplicate_csv_ids = sorted(
            video_id for video_id, count in csv_counts.items() if count > 1
        )
        csv_video_id_set = set(self.csv_video_ids)
        manifest_video_id_set = set(self.video_group_ids)
        if duplicate_csv_ids or csv_video_id_set != manifest_video_id_set:
            missing = sorted(manifest_video_id_set - csv_video_id_set)
            extra = sorted(csv_video_id_set - manifest_video_id_set)
            raise ValueError(
                "train CSV video IDs do not match trusted manifest "
                f"train_video_ids: missing={missing[:5]} extra={extra[:5]} "
                f"duplicates={duplicate_csv_ids[:5]}"
            )

        self.data = json.load(open(json_path, 'r'))     # info videos sentences
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_words_attrs = max_words if max_words_attrs is None else int(max_words_attrs)
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        self.strategy = strategy
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 3: TQFS 帧质量采样
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3]
        _validate_tqfs_dependency(self.slice_framepos)

        self.sample_len = 0
        train_video_ids = set(self.csv_video_ids)
        self.sentences_dict = {}
        caption_counts = Counter()
        for itm in self.data['sentences']:
            if itm['video_id'] in train_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (
                    itm['video_id'], itm['caption']
                )
                caption_counts[itm['video_id']] += 1
        for video_id in self.csv_video_ids:
            count = caption_counts[video_id]
            if count != expected_captions_per_video:
                raise ValueError(
                    f"video_id={video_id} expected "
                    f"{expected_captions_per_video} captions, got {count}"
                )
        self.sample_len = len(self.sentences_dict)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=True, strategy=self.strategy)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.use_attributes = bool(use_attributes)
        self.attributes_path = attributes_path
        self.attributes_map = _load_attributes_map(attributes_path) if self.use_attributes else {}
        self.attr_num_blocks = int(attr_num_blocks) if attr_num_blocks is not None else 4

    def __len__(self):
        return self.sample_len      # copy training dataset

    def _get_text(self, video_id, caption=None, max_words=None):
        k = 1
        choice_video_ids = [video_id]
        mw = self.max_words if max_words is None else int(max_words)
        pairs_text = np.zeros((k, mw), dtype=np.int64)
        pairs_mask = np.zeros((k, mw), dtype=np.int64)
        pairs_segment = np.zeros((k, mw), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)
        
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = mw - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < mw:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == mw
            assert len(input_mask) == mw
            assert len(segment_ids) == mw

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_text_list(self, video_id, captions, max_words=None):
        if not isinstance(captions, (list, tuple)):
            captions = [captions]
        captions = list(captions)
        if len(captions) == 0:
            captions = [""]

        k = len(captions)
        choice_video_ids = [video_id]
        mw = self.max_words if max_words is None else int(max_words)
        pairs_text = np.zeros((k, mw), dtype=np.int64)
        pairs_mask = np.zeros((k, mw), dtype=np.int64)
        pairs_segment = np.zeros((k, mw), dtype=np.int64)

        for i, cap in enumerate(captions):
            if not isinstance(cap, str):
                cap = str(cap)
            words = self.tokenizer.tokenize(cap)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = mw - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < mw:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        # rind = random.randint(0, len(self.sentences[video_id]) - 1)         # randomly choose one single sentence from multi sentences
        rind = len(self.sentences['video_id']) // 2         # fixed selecting the middle (10th) sentence 
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)  # [0]

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)    # 1 12 1 3 224 224

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to its video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            if self.slice_framepos == 3:
                raw_video_data = self.rawVideoExtractor.get_tqfs_video_data(
                    video_path, self.max_frames
                )
            else:
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']    # [max_frames, 3, 224, 224]
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)  # [max_frames, 1, 3, 224, 224]
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:    # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:   # extract uniformly
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                # 帧序：顺序，逆序，随机打乱
                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)     # [max_frames, 1, 3, 224, 224]

                slice_len = video_slice.shape[0]    # max_frames
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawframes(self, choice_video_ids):
        
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float32)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}".format(video_id))  # folder

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:    # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:   # extract uniformly
                        sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_index, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawFramesExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _decode_video_from_path(self, video_path):
        """Deterministic single-video decode reusing the training video extractor.

        Returns ``(video, video_mask)`` shaped ``(max_frames, 1, 3, size, size)``
        / ``(max_frames,)`` — the same shape that :meth:`_get_rawvideo` produces
        for a single-id list.
        """
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros(
            (1, self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32,
        )
        max_video_length = [0]
        if self.slice_framepos == 3:
            raw = self.rawVideoExtractor.get_tqfs_video_data(video_path, self.max_frames)
        else:
            raw = self.rawVideoExtractor.get_video_data(video_path)
        raw = raw["video"]
        if len(raw.shape) <= 3:
            raise RuntimeError(f"CRF decode: empty frames for {video_path}")
        slice_ = self.rawVideoExtractor.process_raw_data(raw)
        if self.max_frames < slice_.shape[0]:
            if self.slice_framepos == 0:
                slice_ = slice_[: self.max_frames]
            elif self.slice_framepos == 1:
                slice_ = slice_[-self.max_frames:]
            else:
                idx = np.linspace(0, slice_.shape[0] - 1, num=self.max_frames, dtype=int)
                slice_ = slice_[idx]
        slice_ = self.rawVideoExtractor.process_frame_order(
            slice_, frame_order=self.frame_order
        )
        slen = slice_.shape[0]
        max_video_length[0] = max(max_video_length[0], slen)
        video[0, :slen, ...] = slice_
        for i, v in enumerate(max_video_length):
            video_mask[i, :v] = 1
        return video[0], video_mask[0]  # drop the outer "pair" axis for collate

    def _get_crf_paired(self, video_id, primary_rank, secondary_rank):
        """CRF-mode paired decode: clean + primary + optional secondary.

        Frame indices are guaranteed identical across the three paths because
        (a) all three MP4s have the same width/height/fps/frame_count (enforced
        by :func:`crf_ladder.codec.assert_transcode_invariants`), and (b) the
        video extractor is deterministic for a given path + settings.
        The plan forbids ``frame_order == 2`` in CRF mode: a random per-call
        permutation would break the paired-index invariant.
        """
        from crf_ladder import crf_output_path

        if self.frame_order == 2:
            raise RuntimeError(
                "CRF training requires deterministic frame_order (0 or 1); "
                "frame_order=2 (random shuffle) breaks the shared-index invariant"
            )
        if self.crf_ladder_root is None:
            raise RuntimeError(
                "MSRVTT_TrainDataLoader received a CRFBatchIndex but "
                "crf_ladder_root was not configured on the dataset"
            )
        primary_crf = RANK_TO_CRF[int(primary_rank)]
        clean_path = os.path.join(self.features_path, f"{video_id}.mp4")
        if not os.path.exists(clean_path):
            clean_path = clean_path.replace(".mp4", ".webm")
        primary_path = str(crf_output_path(self.crf_ladder_root, primary_crf, video_id))
        if not os.path.exists(primary_path):
            raise RuntimeError(
                f"CRF primary missing: video_id={video_id} crf={primary_crf} "
                f"path={primary_path} — refusing to fall back to clean"
            )
        clean_video, clean_mask = self._decode_video_from_path(clean_path)
        primary_video, primary_mask = self._decode_video_from_path(primary_path)
        if primary_video.shape != clean_video.shape:
            raise RuntimeError(
                f"CRF paired decode shape mismatch: video_id={video_id} "
                f"clean={clean_video.shape} primary={primary_video.shape}"
            )
        if not np.array_equal(primary_mask, clean_mask):
            raise RuntimeError(
                f"CRF paired decode mask mismatch: video_id={video_id}"
            )
        secondary_video = None
        secondary_out = 0
        if secondary_rank is not None:
            secondary_crf = RANK_TO_CRF[int(secondary_rank)]
            secondary_path = str(
                crf_output_path(self.crf_ladder_root, secondary_crf, video_id)
            )
            if not os.path.exists(secondary_path):
                raise RuntimeError(
                    f"CRF secondary missing: video_id={video_id} crf={secondary_crf} "
                    f"path={secondary_path} — refusing to fall back to clean"
                )
            secondary_video, secondary_mask = self._decode_video_from_path(secondary_path)
            if secondary_video.shape != clean_video.shape:
                raise RuntimeError(
                    f"CRF paired decode shape mismatch (secondary): video_id={video_id} "
                    f"clean={clean_video.shape} secondary={secondary_video.shape}"
                )
            if not np.array_equal(secondary_mask, clean_mask):
                raise RuntimeError(
                    f"CRF paired decode mask mismatch (secondary): video_id={video_id}"
                )
            secondary_out = int(secondary_rank)
        return clean_video, clean_mask, primary_video, secondary_video, secondary_out

    def __getitem__(self, idx):
        if CRFBatchIndex is not None and isinstance(idx, CRFBatchIndex):
            return self._getitem_crf(idx)
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv_video_ids[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption, max_words=self.max_words)

        if self.use_attributes:
            base_text = caption if caption is not None else " ".join(self._get_single_text(video_id))
            attrs = self.attributes_map.get(video_id, "")
            if not isinstance(attrs, str) or not attrs.strip():
                attrs = base_text  # fallback
            attr_blocks = _split_attr_into_blocks(attrs, num_blocks=self.attr_num_blocks)
            pairs_text_a, pairs_mask_a, pairs_segment_a, _ = self._get_text_list(
                video_id, attr_blocks, max_words=self.max_words_attrs
            )
        else:
            pairs_text_a = pairs_mask_a = pairs_segment_a = None

        video, video_mask = self._get_rawvideo(choice_video_ids)
        # video, video_mask = self._get_rawframes(choice_video_ids)
        video_group_id = np.int64(self.video_group_ids[video_id])
        if self.use_attributes:
            return (
                pairs_text,
                pairs_mask,
                pairs_segment,
                pairs_text_a,
                pairs_mask_a,
                pairs_segment_a,
                video,
                video_mask,
                video_group_id,
            )
        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            video,
            video_mask,
            video_group_id,
        )

    def _getitem_crf(self, idx):
        """CRF-mode sample: dict consumed by ``crf_pairing.crf_train_collate``.

        Text comes from the standard sentence path so retrieval loss stays on
        the same distribution as baseline runs. Videos come from the paired
        decode: clean is always present; primary is the CRF at
        ``RANK_TO_CRF[idx.primary_rank]``; secondary is present iff
        ``idx.secondary_rank`` is not None. All three share frame indices.
        """
        video_id, caption = self.sentences_dict[idx.index]
        pairs_text, pairs_mask, pairs_segment, _ = self._get_text(
            video_id, caption, max_words=self.max_words
        )
        clean_video, video_mask, primary_video, secondary_video, secondary_rank = (
            self._get_crf_paired(video_id, idx.primary_rank, idx.secondary_rank)
        )
        return {
            "input_ids": pairs_text[0],
            "input_mask": pairs_mask[0],
            "segment_ids": pairs_segment[0],
            "clean_video": clean_video,
            "video_mask": video_mask,
            "group_ids": np.int64(self.video_group_ids[video_id]),
            "degraded_video": primary_video,
            "primary_rank": np.int64(idx.primary_rank),
            "secondary_video": secondary_video,
            "secondary_rank": secondary_rank,
        }


