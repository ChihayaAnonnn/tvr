from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import pickle
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

from dataloaders.rawvideo_util import RawVideoExtractor


def _split_attr_into_blocks(text: str, num_blocks: int = 4):
    """Split Qwen-generated attributes into semantic blocks.

    Expected format contains headings like:
      【ENTITIES】, 【ACTIONS】, 【APPEARANCE & DETAILS】, 【SCENE】, 【TEXT/OCR】

    Returns a list of block strings (length==num_blocks). Missing blocks will be padded with "".
    For num_blocks=4, APPEARANCE will be merged into ENTITIES.
    """
    num_blocks = int(num_blocks) if num_blocks is not None else 4
    if num_blocks <= 0:
        num_blocks = 1
    if not isinstance(text, str):
        return [""] * num_blocks
    s = text.strip()
    if not s:
        return [""] * num_blocks

    headings = [
        "【ENTITIES】",
        "【ACTIONS】",
        "【APPEARANCE & DETAILS】",
        "【SCENE】",
        "【TEXT/OCR】",
    ]
    pos = []
    for h in headings:
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

    # IMPORTANT: keep a fixed number of blocks for DataLoader collation (k must be constant).
    blocks = [(b if isinstance(b, str) else "") for b in blocks]
    if len(blocks) < num_blocks:
        blocks = blocks + [""] * (num_blocks - len(blocks))
    return blocks[:num_blocks]


def _load_attributes_map(path: str) -> Dict[str, str]:
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
    # If a directory is provided, caller should pass a concrete file path.
    if os.path.isdir(path):
        return {}
    if not os.path.exists(path):
        print(f"[MSVD] attributes_path not found: {path}")
        return {}

    if path.endswith(".jsonl"):
        m: Dict[str, str] = {}
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
        print(f"[MSVD] Loaded attributes map from jsonl: {path}, size={len(m)}")
        return m

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        m = {k: v for k, v in obj.items() if isinstance(k, str) and isinstance(v, str) and v.strip()}
        print(f"[MSVD] Loaded attributes map from json: {path}, size={len(m)}")
        return m
    print(f"[MSVD] Unexpected attributes file schema: {path}")
    return {}

class MSVD_DataLoader(Dataset):
    """MSVD dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
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
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_words_attrs = max_words if max_words_attrs is None else int(max_words_attrs)
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.use_attributes = bool(use_attributes)
        self.attributes_path = attributes_path
        self.attributes_map = _load_attributes_map(attributes_path) if self.use_attributes else {}
        self.attr_num_blocks = int(attr_num_blocks) if attr_num_blocks is not None else 4

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption, max_words=None):
        k = 1
        choice_video_ids = [video_id]
        mw = self.max_words if max_words is None else int(max_words)
        pairs_text = np.zeros((k, mw), dtype=int)
        pairs_mask = np.zeros((k, mw), dtype=int)
        pairs_segment = np.zeros((k, mw), dtype=int)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

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
        """Tokenize a list of captions into (k, mw) arrays."""
        if not isinstance(captions, (list, tuple)):
            captions = [captions]
        captions = list(captions)
        if len(captions) == 0:
            captions = [""]

        k = len(captions)
        choice_video_ids = [video_id]
        mw = self.max_words if max_words is None else int(max_words)
        pairs_text = np.zeros((k, mw), dtype=int)
        pairs_mask = np.zeros((k, mw), dtype=int)
        pairs_segment = np.zeros((k, mw), dtype=int)

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

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

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

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption, max_words=self.max_words)

        if self.use_attributes:
            attrs = self.attributes_map.get(video_id, "")
            if not isinstance(attrs, str) or not attrs.strip():
                attrs = caption  # fallback
            attr_blocks = _split_attr_into_blocks(attrs, num_blocks=self.attr_num_blocks)
            pairs_text_a, pairs_mask_a, pairs_segment_a, _ = self._get_text_list(
                video_id, attr_blocks, max_words=self.max_words_attrs
            )
        else:
            pairs_text_a = pairs_mask_a = pairs_segment_a = None

        video, video_mask = self._get_rawvideo(choice_video_ids)
        if self.use_attributes:
            return pairs_text, pairs_mask, pairs_segment, pairs_text_a, pairs_mask_a, pairs_segment_a, video, video_mask
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
