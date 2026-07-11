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
from dataloaders.hard_negative_mapping import load_hard_negative_index
from dataloaders.msrvtt_protocol import load_trusted_manifest
from dataloaders.rawframes_util import RawFramesExtractor
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.tqfs_cache import TQFSFrameCache


def _validate_tqfs_dependency(slice_framepos):
    if slice_framepos == 3 and importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "slice_framepos=3 requires scikit-learn for deterministic TQFS; "
            "install the pinned project requirements instead of silently falling back"
        )


def _build_tqfs_cache(
    cache_dir,
    slice_framepos,
    features_path,
    feature_framerate,
    max_frames,
    image_resolution,
):
    if slice_framepos != 3 or not cache_dir:
        return None
    return TQFSFrameCache(
        cache_dir,
        features_path=features_path,
        feature_framerate=feature_framerate,
        max_frames=max_frames,
        image_resolution=image_resolution,
    )


def _get_tqfs_video_data(extractor, cache, video_id, video_path, max_frames):
    if cache is not None:
        cached = cache.load(video_id)
        if cached is not None:
            return {"video": cached}

    result = extractor.get_tqfs_video_data(video_path, max_frames)
    video = result["video"]
    if cache is not None and getattr(video, "ndim", 0) == 4:
        cache.store(video_id, video)
    return result


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
            tqfs_cache_dir="",
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
        self.tqfs_cache = _build_tqfs_cache(
            tqfs_cache_dir,
            self.slice_framepos,
            features_path,
            feature_framerate,
            max_frames,
            image_resolution,
        )

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
                raw_video_data = _get_tqfs_video_data(
                    self.rawVideoExtractor,
                    self.tqfs_cache,
                    video_id,
                    video_path,
                    self.max_frames,
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
            return_sample_index=False,
            return_hard_negative=False,
            hard_negative_path="",
            split_manifest_path=None,
            tqfs_cache_dir="",
            expected_captions_per_video=20,
    ):
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
        self.tqfs_cache = _build_tqfs_cache(
            tqfs_cache_dir,
            self.slice_framepos,
            features_path,
            feature_framerate,
            max_frames,
            image_resolution,
        )
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.use_attributes = bool(use_attributes)
        self.attributes_path = attributes_path
        self.attributes_map = _load_attributes_map(attributes_path) if self.use_attributes else {}
        self.attr_num_blocks = int(attr_num_blocks) if attr_num_blocks is not None else 4
        self.return_hard_negative = bool(return_hard_negative)
        if return_sample_index and not self.return_hard_negative:
            raise ValueError(
                "sample_index is only available with explicit hard negatives"
            )
        self.hard_index = []
        if self.return_hard_negative:
            if not self.unfold_sentences:
                raise ValueError("Explicit hard-negative training requires unfold_sentences=True")
            self.hard_index = load_hard_negative_index(hard_negative_path, self.sample_len)

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
                raw_video_data = _get_tqfs_video_data(
                    self.rawVideoExtractor,
                    self.tqfs_cache,
                    video_id,
                    video_path,
                    self.max_frames,
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

    def _get_hard_negative_video(self, anchor_video_id, idx):
        hard_idx = -1
        if 0 <= int(idx) < len(getattr(self, "hard_index", [])):
            hard_idx = int(self.hard_index[int(idx)])

        valid = np.int64(0)
        hard_video_id = anchor_video_id
        if hard_idx >= 0 and hasattr(self, "sentences_dict") and hard_idx in self.sentences_dict:
            hard_video_id = self.sentences_dict[hard_idx][0]
            valid = np.int64(1)

        hard_video, hard_video_mask = self._get_rawvideo([hard_video_id])
        return hard_video, hard_video_mask, valid


    def __getitem__(self, idx):
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
        sample_index = np.int64(idx)
        video_group_id = np.int64(self.video_group_ids[video_id])
        if self.return_hard_negative:
            hard_video, hard_video_mask, hard_valid = self._get_hard_negative_video(video_id, idx)
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
                    sample_index,
                    hard_video,
                    hard_video_mask,
                    hard_valid,
                    video_group_id,
                )
            return (
                pairs_text,
                pairs_mask,
                pairs_segment,
                video,
                video_mask,
                sample_index,
                hard_video,
                hard_video_mask,
                hard_valid,
                video_group_id,
            )
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


