"""Hash-verified, pre-temporal frozen frame features."""
from dataclasses import asdict, dataclass
import os
from pathlib import Path

import torch

from experiment_artifacts import artifact_json_bytes, strict_json_load, verify_array_reference
from semantic_retrieval.evidence.types import digest, require_digest, positive


@dataclass(frozen=True)
class FrameCacheIdentity:
    content_sha256: str
    clip_sha256: str
    clip_config_sha256: str
    preprocessing_sha256: str

    def __post_init__(self):
        for value in asdict(self).values():
            require_digest(value)


def safe_path(path):
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError('symlink in evidence path')


def publish_json(path, payload):
    safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = artifact_json_bytes(payload)
    if len(data) >= 1048576:
        raise ValueError('externalize large evidence payload')
    tmp = path.with_name(path.name+'.tmp')
    safe_path(tmp)
    with tmp.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FrameFeatureCache:
    def __init__(self, store):
        self.store = store
        self.root = store.run_root/'provenance'/'frame-cache'
        safe_path(self.root)

    def _path(self, candidate_id, frame_id, identity):
        positive(frame_id, integer=True, zero=True)
        if not candidate_id:
            raise ValueError('candidate identity required')
        key = digest(dict(candidate=candidate_id, frame=frame_id, identity=asdict(identity)))
        p = self.root/(key+'.json'); safe_path(p)
        return p

    def write(self, candidate_id, frame_id, feature, *, identity):
        if feature.ndim != 1 or not feature.is_floating_point() or feature.requires_grad or not torch.isfinite(feature).all():
            raise ValueError('cache requires detached finite floating feature vector')
        p = self._path(candidate_id, frame_id, identity)
        if p.exists():
            old = self.read(candidate_id, frame_id, identity=identity)
            if not torch.equal(old, feature.cpu()):
                raise ValueError('conflicting frame cache contents')
            with p.open() as h:
                return strict_json_load(h)['reference']
        ref = self.store.write('frame_'+p.stem, feature.detach().cpu())
        publish_json(p, dict(identity=asdict(identity), candidate=candidate_id, frame=frame_id, reference=asdict(ref)))
        return ref

    def read(self, candidate_id, frame_id, *, identity):
        p = self._path(candidate_id, frame_id, identity)
        if not p.exists():
            raise ValueError('frame cache identity not present')
        with p.open() as h:
            value = strict_json_load(h)
        if value['identity'] != asdict(identity) or value['candidate'] != candidate_id or value['frame'] != frame_id:
            raise ValueError('frame cache identity mismatch')
        return torch.from_numpy(verify_array_reference(self.store.run_root, value['reference']).copy())
