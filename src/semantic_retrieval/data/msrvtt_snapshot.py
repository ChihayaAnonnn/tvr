"""Allowlist-first extraction: sealed caption values remain encoded bytes."""

import hashlib
import json
import re
from collections import Counter

TOKEN = re.compile(
    rb'[ \t\r\n]*("(?:[^"\\]|\\.)*"|[{}\[\],:]|true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
)


class Cursor:
    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def token(self):
        match = TOKEN.match(self.raw, self.pos)
        if match is None:
            raise ValueError("invalid JSON token or unexpected end")
        self.pos = match.end()
        return match.group(1)

    def expect(self, expected):
        if self.token() != expected:
            raise ValueError("unexpected JSON delimiter")

    def value(self):
        begin = self.pos
        first = self.token()
        if first in (b"{", b"["):
            stack = [b"}" if first == b"{" else b"]"]
            while stack:
                token = self.token()
                if token in (b"{", b"["):
                    stack.append(b"}" if token == b"{" else b"]")
                elif token in (b"}", b"]"):
                    if token != stack.pop():
                        raise ValueError("mismatched JSON delimiters")
        return self.raw[begin : self.pos].strip()


def record_fields(raw):
    c = Cursor(raw)
    c.expect(b"{")
    fields = {}
    token = c.token()
    if token == b"}":
        return fields
    while True:
        if not token.startswith(b'"'):
            raise ValueError("object key must be a string")
        key = json.loads(token)  # Decode field names, never caption values here.
        if key in fields:
            raise ValueError("duplicate record key")
        c.expect(b":")
        fields[key] = c.value()
        token = c.token()
        if token == b"}":
            break
        if token != b",":
            raise ValueError("invalid object delimiter")
        token = c.token()
    return fields


def training_annotations(raw, allowed, *, decode_caption=json.loads):
    if not allowed or any(not isinstance(i, str) or not i for i in allowed):
        raise ValueError("training allowlist is required")
    cursor = Cursor(raw)
    cursor.expect(b"{")
    rows = []
    seen = set()
    found = False
    row_index = 0
    counts = dict(
        selected_records=0, excluded_records=0, decoded_training_caption_values=0, decoded_excluded_caption_values=0
    )
    token = cursor.token()
    while token != b"}":
        key = json.loads(token)
        cursor.expect(b":")
        if key != "sentences":
            cursor.value()
        else:
            if found:
                raise ValueError("duplicate sentence collection")
            found = True
            cursor.expect(b"[")
            while True:
                saved = cursor.pos
                token = cursor.token()
                if token == b"]":
                    break
                cursor.pos = saved
                fields = record_fields(cursor.value())
                row_index += 1
                if "video_id" not in fields or "caption" not in fields:
                    raise ValueError("missing official caption/video relationship")
                official = json.loads(fields["video_id"])
                if not isinstance(official, str) or not official:
                    raise ValueError("invalid official video identity")
                if official in allowed:
                    caption = decode_caption(fields["caption"])
                    if not isinstance(caption, str) or not caption.strip():
                        raise ValueError("invalid training caption")
                    caption_id = json.loads(fields["sen_id"]) if "sen_id" in fields else f"row-{row_index:09}"
                    if type(caption_id) not in (str, int) or not str(caption_id):
                        raise ValueError("invalid immutable caption identity")
                    caption_id = str(caption_id)
                    query_id = f"{official}:{caption_id}"
                    if query_id in seen:
                        raise ValueError("duplicate official caption identity")
                    seen.add(query_id)
                    rows.append(
                        dict(query_id=query_id, caption_id=caption_id, official_video_id=official, query_text=caption)
                    )
                    counts["selected_records"] += 1
                    counts["decoded_training_caption_values"] += 1
                else:
                    counts["excluded_records"] += 1
                token = cursor.token()
                if token == b"]":
                    break
                if token != b",":
                    raise ValueError("invalid sentence delimiter")
        token = cursor.token()
        if token == b"}":
            break
        if token != b",":
            raise ValueError("invalid root delimiter")
        token = cursor.token()
    if not found or raw[cursor.pos :].strip():
        raise ValueError("missing sentences or trailing JSON bytes")
    return rows, counts


def caption_relation_digest(rows):
    by_video = {}
    for row in rows:
        by_video.setdefault(row["official_video_id"], Counter())[row["query_text"]] += 1
    digest = hashlib.sha256()
    for video in sorted(by_video):
        for text, count in sorted(by_video[video].items()):
            digest.update(json.dumps([video, text, count], ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def inspect_training_video(path, *, expected_sha256, expected_frame_count=None, image_size=224, max_frames=8):
    """Count every decoded source frame, then validate requested uniform samples."""
    import cv2

    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.data.video import decode_video, sample_frame_indices

    if file_sha256(path) != expected_sha256:
        raise ValueError("training video content changed")
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("training video cannot be opened")
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count < 1 or expected_frame_count is not None and count != expected_frame_count:
        raise ValueError("decoded source frame count mismatch")
    _, valid = decode_video(path, sample_frame_indices(count, max_frames), image_size)
    if not valid.all():
        raise ValueError("uniform source frames are not all decodable")
    return count
