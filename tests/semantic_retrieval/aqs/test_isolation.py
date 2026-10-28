import ast
import os
from pathlib import Path
import subprocess
import sys


def test_source_dependency_boundary():
    root = Path(__file__).resolve().parents[3]
    for path in (root/'src/semantic_retrieval/aqs').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom): names = [node.module or '']
            else: continue
            assert not any('laya' in n.lower() or 'querymatch' in n.lower() for n in names), path


def test_fresh_process_runs_without_external_reranker_or_cache(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = r'''
import os, sys, pathlib, importlib.abc
scratch = pathlib.Path(sys.argv[1]).resolve()
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        name = fullname.lower()
        if 'laya' in name or 'querymatch' in name or name.startswith('transformers'):
            raise ImportError('external reranker unavailable')
sys.meta_path.insert(0, BlockExternal())
def audit(event, args):
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        text = str(path).lower()
        if 'laya' in text or 'querymatch' in text:
            raise PermissionError('external reranker path unavailable')
        if event == 'open':
            mode = args[1] or ''
            flags = args[2] if len(args)>2 else 0
            writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and path != scratch and scratch not in path.parents and str(path) != '/dev/null':
                raise PermissionError('test writes must stay inside scratch: '+str(path))
sys.addaudithook(audit)
try:
    __import__('semantic_retrieval.laya_rerank')
except ImportError:
    pass
else:
    raise AssertionError('import blocker did not work')
try:
    open(scratch/'artifacts/laya-rerank-validation-v1/forbidden', 'rb')
except PermissionError:
    pass
else:
    raise AssertionError('path blocker did not work')
import torch
from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.model import AqsBaselineModel
from semantic_retrieval.aqs.training import video_id_mapping
torch.set_num_threads(1)
torch.manual_seed(713)
clip = ClipBackbone(ClipConfig(64,16,64,1,4,32,128,64,1,1,fixture=True))
_, digest = video_id_mapping(['v0','v1'])
model = AqsBaselineModel(clip.config, clip.state_dict(), aqs=AqsConfig(True,2,.2), id_mapping_sha256=digest, temporal_layers=1)
ids = torch.tensor([[126,4,127,0],[126,5,127,0]])
video = torch.randn(2,3,3,16,16)
batch = (ids, ids!=0, video, torch.ones(2,3,dtype=torch.bool), torch.arange(2))
model(*batch)['loss'].backward()
disabled = AqsBaselineModel(clip.config, clip.state_dict(), aqs=AqsConfig(False,2,.2), id_mapping_sha256=digest, temporal_layers=1)
disabled(*batch)['loss'].backward()
model.eval()
with torch.no_grad():
    result = model.score(model.encode_text(*batch[:2]), model.encode_video(*batch[2:4]))
assert result.shape == (2,2) and torch.isfinite(result).all()
assert not any('laya' in name.lower() or 'querymatch' in name.lower() for name in sys.modules)
print('isolated ok')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join((str(root/'src'), str(root))), TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'isolated ok' in result.stdout
