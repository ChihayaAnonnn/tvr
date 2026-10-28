"""Hash-verified full video feature cache backed by run-local arrays."""

import hashlib
import json
import re
from dataclasses import asdict

import torch

from experiment_artifacts import artifact_json_bytes, verify_array_reference
from semantic_retrieval.types import VideoFeatures


class FeatureCache:
    def __init__(self, context):
        self.store = context.array_store

    def _path(self, canonical_id):
        if not isinstance(canonical_id, str) or not canonical_id:
            raise ValueError("invalid canonical ID")
        return self.store.root / ("feature-" + hashlib.sha256(canonical_id.encode()).hexdigest() + ".json")

    @staticmethod
    def _validate(features):
        f, p, m = features.frame_tokens, features.pooled_video, features.frame_mask
        if (
            f.ndim != 3
            or f.shape[0] != 1
            or p.shape != (1, f.shape[-1])
            or m.shape != f.shape[:2]
            or m.dtype != torch.bool
            or not m.any(-1).all()
            or not torch.isfinite(f).all()
            or not torch.isfinite(p).all()
        ):
            raise ValueError("invalid cached feature schema")

    def write(self, canonical_id, features, *, identity, content_sha256):
        self._validate(features)
        if any(x.requires_grad for x in (features.frame_tokens, features.pooled_video)):
            raise ValueError("cache only frozen features")
        if not re.fullmatch("[0-9a-f]{64}", content_sha256):
            raise ValueError("invalid content hash")
        arrays = {
            name: self.store.write(name, getattr(features, name).detach().cpu().numpy()).as_dict()
            for name in ("frame_tokens", "pooled_video", "frame_mask")
        }
        payload = dict(
            schema_version=1,
            canonical_id=canonical_id,
            identity=asdict(identity),
            content_sha256=content_sha256,
            arrays=arrays,
        )
        path = self._path(canonical_id)
        encoded = artifact_json_bytes(payload)
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError("immutable cache entry conflict")
        else:
            with path.open("xb") as handle:
                handle.write(encoded)

    def read(self, canonical_id, *, expected_identity, content_sha256):
        payload = json.loads(self._path(canonical_id).read_text())
        if (
            set(payload) != {"schema_version", "canonical_id", "identity", "content_sha256", "arrays"}
            or payload["schema_version"] != 1
            or payload["canonical_id"] != canonical_id
            or payload["identity"] != asdict(expected_identity)
            or payload["content_sha256"] != content_sha256
        ):
            raise ValueError("cache identity mismatch")
        if set(payload["arrays"]) != {"frame_tokens", "pooled_video", "frame_mask"}:
            raise ValueError("cache schema mismatch")
        f = VideoFeatures(
            **{
                name: torch.from_numpy(verify_array_reference(self.store.run_root, ref).copy())
                for name, ref in payload["arrays"].items()
            }
        )
        self._validate(f)
        return f
