"""Immutable content-bound array cache. Each directory has one writer."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic(path, raw):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


class ArrayCache:
    def __init__(self, root, identity):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        path = self.root/'identity.json'
        raw = canonical(identity)
        if path.exists():
            if path.read_bytes() != raw: raise ValueError('cache identity mismatch')
        else:
            if any(self.root.iterdir()): raise ValueError('unidentified nonempty cache')
            atomic(path, raw)

    def paths(self, key):
        name = hashlib.sha256(canonical({'identity': self.identity, 'key': key})).hexdigest()
        return self.root/(name+'.npy'), self.root/(name+'.json')

    def read(self, key):
        path, marker = self.paths(key)
        if not marker.exists(): return None  # Uncommitted array is not a completed block.
        rec = json.loads(marker.read_text())
        if rec['key'] != key or not path.is_file() or digest(path) != rec['sha256']:
            raise ValueError('cache content/identity corruption')
        a = np.load(path, allow_pickle=False)
        if list(a.shape) != rec['shape'] or str(a.dtype) != rec['dtype'] or not np.isfinite(a).all():
            raise ValueError('cache schema corruption')
        return a

    def write(self, key, array):
        a = np.asarray(array)
        if a.dtype.kind not in 'fiu' or not np.isfinite(a).all(): raise ValueError('invalid cache array')
        old = self.read(key)
        if old is not None:
            if old.dtype != a.dtype or not np.array_equal(old, a): raise ValueError('immutable cache overwrite')
            return
        path, marker = self.paths(key)
        fd, name = tempfile.mkstemp(dir=self.root, prefix='.pending-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, a, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, path)
            atomic(marker, canonical(dict(key=key, sha256=digest(path), shape=list(a.shape), dtype=str(a.dtype))))
        finally:
            if os.path.exists(name): os.unlink(name)
