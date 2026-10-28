"""CPU-resident query-independent visual tokens, persisted with content hashes."""
import numpy as np
import torch
from .store import ArrayCache


class VisualCache:
    def __init__(self, root, identity):
        self.store=ArrayCache(root,identity)
        self.memory={}

    def get(self, key, images, judge):
        from .store import canonical
        encoded=canonical(key)
        if encoded in self.memory: return self.memory[encoded]
        features=self.store.read({'video':key,'kind':'features'})
        ids=self.store.read({'video':key,'kind':'prefix_ids'})
        if features is None or ids is None:
            value=judge.encode(images())
            features=value['features'].numpy()
            ids=np.array(value['ids'],dtype=np.int64)
            self.store.write({'video':key,'kind':'features'},features)
            self.store.write({'video':key,'kind':'prefix_ids'},ids)
        if features.dtype!=np.float32 or ids.dtype!=np.int64 or ids.ndim!=1 or features.ndim!=3:
            raise ValueError('invalid cached visual token schema')
        value={'features':torch.from_numpy(features),'ids':ids.tolist()}
        self.memory[encoded]=value
        return value
