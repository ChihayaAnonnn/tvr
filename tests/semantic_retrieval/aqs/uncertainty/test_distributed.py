import copy
import datetime
from pathlib import Path
import pytest
import torch
import torch.distributed as dist


class Objective(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,*batch):
        out=self.model(*batch)
        return out['contrastive']+.2*out['aqs'],out['logits'].detach()


def worker(rank,init_file,directory,config,initial,batch,mode):
    from semantic_retrieval.aqs.fusion_model import FusionAqsModel
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    from semantic_retrieval.aqs.config import AqsConfig
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=30))
    path=Path(directory)/f'{rank}.pt'
    clip={k.removeprefix('core.clip.'):v for k,v in initial.items() if k.startswith('core.clip.')}
    try:
        try:
            model=FusionAqsModel(config,clip,aqs=AqsConfig(True,2,.2),fusion=FusionConfig(mode!='late_aqs',.2 if mode=='eta' and rank else .1),
                                 id_mapping_sha256='a'*64,temporal_layers=1)
            model.load_state_dict(initial,strict=True)
            local=tuple(x[rank*2:rank*2+2].clone() for x in batch)
            if mode=='duplicate' and rank: local[-1][0]=0
            if mode=='mask' and rank: local[3][0]=False
            if mode=='late_aqs' and rank:model.aqs_config=AqsConfig(True,2,0.)
            if mode in ('resume','optimizer'):
                from semantic_retrieval.aqs.fusion_training import fusion_step,save_fusion_state,restore_fusion_state
                from semantic_retrieval.full_data.training import make_optimizer
                from .test_resume import binding_for,fingerprint
                from experiment_artifacts import ArtifactArrayStore
                opt=make_optimizer(model,learning_rate=5e-5,total_steps=10)
                if mode=='optimizer' and rank:
                    group=opt.param_groups[0]
                    group['params'][0]=torch.nn.Parameter(group['params'][0].detach().clone())
                ddp=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
                first=fusion_step(model,ddp,opt,local)
                root=Path(directory)/'resume';(root/'checkpoints').mkdir(parents=True,exist_ok=True)
                checkpoint=root/'checkpoints/one.pt';binding=binding_for(model,world=2)
                save_fusion_state(checkpoint,model,opt,binding,dict(epoch=0,batch_cursor=1,update=1),ArtifactArrayStore(root/'arrays'))
                expected=fusion_step(model,ddp,opt,local);end=fingerprint(model,opt)
                fresh=FusionAqsModel(config,clip,aqs=AqsConfig(True,2,.2),fusion=FusionConfig(True,.1),id_mapping_sha256='a'*64,temporal_layers=1)
                fresh_opt=make_optimizer(fresh,learning_rate=5e-5,total_steps=10)
                fresh_ddp=torch.nn.parallel.DistributedDataParallel(fresh,find_unused_parameters=True)
                restore_fusion_state(checkpoint,fresh,fresh_opt,binding)
                assert fusion_step(fresh,fresh_ddp,fresh_opt,local)==expected
                assert fingerprint(fresh,fresh_opt)==end
                torch.save(dict(resumed=True),path);return
            ddp=torch.nn.parallel.DistributedDataParallel(Objective(model),find_unused_parameters=True)
            if mode=='probability' and rank:
                with torch.no_grad():model.core.uncertain_net_video.fc.weight.fill_(float('nan'))
            loss,logits=ddp(*local);loss.backward()
            if mode!='normal':raise AssertionError('expected collective refusal')
            torch.save(dict(loss=loss.detach(),logits=logits,grads={k:p.grad for k,p in model.named_parameters() if p.grad is not None}),path)
        except (ValueError,RuntimeError) as exc:
            if mode in ('normal','resume'):raise
            # A timeout is not evidence of a collective input rejection.
            assert 'timeout' not in str(exc).lower() and 'timed out' not in str(exc).lower(),str(exc)
            import re
            expected=dict(eta='configuration',duplicate='unique',mask='mask|padding|frame',
                          probability='Gaussian',late_aqs='configuration',optimizer='optimizer parameters')[mode]
            assert re.search(expected,str(exc)),str(exc)
            torch.save(dict(refused=True,error=str(exc)),path)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('mode',['normal','resume','eta','duplicate','mask','probability','late_aqs','optimizer'])
def test_global_gradient_resume_or_refusal(tmp_path,tiny_clip,paired_batch,fusion_factory,mode):
    model=fusion_factory();initial=copy.deepcopy(model.state_dict())
    batch=tuple(torch.cat([x,x]) for x in paired_batch[:-1])+(torch.arange(4),)
    batch[0][2,1]=8;batch[0][3,1]=9;batch[2][2:]+=.1
    out=model(*batch);expected=out['contrastive']+.2*out['aqs'];expected.backward()
    grads={k:p.grad for k,p in model.named_parameters() if p.grad is not None}
    context=torch.multiprocessing.spawn(worker,args=(str(tmp_path/'group'),str(tmp_path),tiny_clip.config,initial,batch,mode),nprocs=2,join=False)
    import time
    deadline=time.monotonic()+65
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise AssertionError('distributed test exceeded deadline')
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        result=torch.load(tmp_path/f'{rank}.pt',weights_only=True)
        if mode=='resume':assert result['resumed'];continue
        if mode!='normal':assert result['refused'];continue
        torch.testing.assert_close(result['loss'],expected,atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(result['logits'],out['logits'],atol=2e-6,rtol=2e-4)
        assert result['grads'].keys()==grads.keys()
        for k,g in grads.items():torch.testing.assert_close(result['grads'][k],g,atol=2e-6,rtol=2e-4,msg=k)
