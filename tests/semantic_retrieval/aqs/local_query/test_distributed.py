import copy
import datetime
import time
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.fusion_config import FusionConfig
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.local_model import build_local_model


def worker(rank, group, directory, config, clip, initial, batch, mode):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+group,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=25))
    try:
        try:
            model=build_local_model(config,clip,local=LocalQueryConfig(True,.3 if mode=='config' and rank else .2,.1),
                fusion=FusionConfig(False,0),aqs=AqsConfig(True,1,.2),id_mapping_sha256='b'*64 if mode=='ids' and rank else 'a'*64,
                temporal_layers=1,samples=2,query_chunk=1,video_chunk=1)
            model.load_state_dict(initial)
            local=tuple(x[rank*2:rank*2+2].clone() for x in batch)
            if mode=='duplicate' and rank: local[-1][0]=0
            if mode=='empty' and rank:
                local[0][0]=0;local[0][0,:2]=torch.tensor([126,127]);local[1][0]=False;local[1][0,:2]=True
            if mode=='nonfinite' and rank: local[2][0,0,0,0,0]=float('nan')
            if mode=='shape' and rank:local=local[:2]+(local[2][:,:2],local[3][:,:2],local[4])
            network=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
            torch.manual_seed(99)
            out=network(*local);out['loss'].backward()
            if mode!='normal': raise AssertionError('expected rejection')
            torch.save(dict(loss=out['loss'].detach(),logits=out['logits'].detach(),
                grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}),Path(directory)/f'{rank}.pt')
        except (ValueError,RuntimeError) as exc:
            if mode=='normal':raise
            assert 'timed out' not in str(exc).lower() and 'timeout' not in str(exc).lower()
            expected={'config':'configuration','ids':'mapping','duplicate':'unique','empty':'lexical','nonfinite':'finite','shape':'shape'}[mode]
            assert expected in str(exc),str(exc)
            torch.save(dict(refused=True),Path(directory)/f'{rank}.pt')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('mode',['normal','config','ids','duplicate','empty','nonfinite','shape'])
def test_distributed_gradient_and_collective_rejection(tmp_path,local_clip,local_batch,mode):
    from .test_model import build
    model=build(local_clip)
    initial=copy.deepcopy(model.state_dict())
    batch=tuple(torch.cat([x,x]) for x in local_batch[:-1])+(torch.arange(4),)
    batch[0][2,1]=8;batch[0][3,1]=9;batch[2][2:]+=.1
    torch.manual_seed(99)
    expected=model(*batch);expected['loss'].backward()
    grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
    ctx=torch.multiprocessing.spawn(worker,args=(str(tmp_path/'group'),str(tmp_path),local_clip.config,
        local_clip.state_dict(),initial,batch,mode),nprocs=2,join=False)
    deadline=time.monotonic()+55
    try:
        while not ctx.join(timeout=1):
            if time.monotonic()>deadline:raise AssertionError('distributed deadline exceeded')
    finally:
        for p in ctx.processes:
            if p.is_alive():p.terminate()
            p.join(timeout=5)
    for rank in range(2):
        got=torch.load(tmp_path/f'{rank}.pt',weights_only=True)
        if mode!='normal': assert got['refused'];continue
        torch.testing.assert_close(got['loss'],expected['loss'],atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(got['logits'],expected['logits'],atol=2e-6,rtol=2e-4)
        assert got['grads'].keys()==grads.keys()
        for n,g in grads.items():torch.testing.assert_close(got['grads'][n],g,atol=2e-6,rtol=2e-4,msg=n)
