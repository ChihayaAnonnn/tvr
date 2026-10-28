"""CPU-only, provenance-bound implementation audit; never trains/selects on test."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
CONTROL = Path('experiments/full-data-implementation-audit-v1')


def compare_arrays(a, b, *, atol=2e-6, rtol=2e-4):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape or finite contract failed')
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(passed=bool(np.allclose(a, b, atol=atol, rtol=rtol)),
                max_abs=float(delta.max(initial=0)),
                relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(b.ravel()), 1e-12)),
                atol=atol, rtol=rtol)


def loss_summary(losses, updates):
    import numpy as np
    if losses.shape != (updates, 4) or not np.isfinite(losses).all():
        raise ValueError('loss updates/shape/finite mismatch')
    keys = ('loss', 'contrastive', 'mil', 'kl')
    return dict(mean=dict(zip(keys, map(float, losses.mean(0)))),
                first_100=dict(zip(keys, map(float, losses[:100].mean(0)))),
                last_100=dict(zip(keys, map(float, losses[-100:].mean(0)))),
                component_sum_max_error=float(np.abs(losses[:, 0]-losses[:, 1:].sum(1)).max()))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self, manifest):
        from semantic_retrieval.runtime.input_bindings import manifest_input_references
        self.references = manifest_input_references(ROOT, manifest)
        self.verified = set()

    def check(self, path):
        p = Path(path).resolve()
        if str(p) not in self.references:
            raise ValueError('unbound audit input: '+str(p))
        if p not in self.verified:
            if digest(p) != self.references[str(p)]: raise ValueError('audit input changed: '+str(p))
            self.verified.add(p)
        return p

    def read(self, path):
        from experiment_artifacts import strict_json_load
        with self.check(path).open() as f: return strict_json_load(f)

    def tensor(self, value, root):
        import torch
        from experiment_artifacts import verify_array_reference
        if isinstance(value, dict) and set(value) == {'__array_reference__'}:
            ref = value['__array_reference__']; self.check(root/ref['path'])
            return torch.from_numpy(verify_array_reference(root, ref).copy())
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('invalid checkpoint model tensor')
        return value


def publish(path, value):
    from semantic_retrieval.evidence.frame_cache import publish_json
    if path.exists(): raise FileExistsError(path)
    publish_json(path, value)


def trajectory(inputs, config, initial):
    import numpy as np
    import torch
    import re
    from experiment_artifacts import verify_array_reference, manifest_sha256
    from semantic_retrieval.full_data.upstream_uatvr.nn_components.optimization import warmup_cosine
    records = []; registry = inputs.read(config['training_registry_snapshot'])
    for parent in config['parents']:
        root = Path(parent['root']); manifest = inputs.read(root/'manifest.json')
        status = inputs.read(root/'status.json')
        if status['state'] != 'completed' or manifest_sha256(manifest) != parent['manifest_sha256']:
            raise ValueError('invalid completed training parent')
        if not any(r['event'] == 'completed' and r['run_id'] == manifest['run_id']
                   and r['manifest_sha256'] == parent['manifest_sha256'] for r in registry):
            raise ValueError('training terminal not bound')
        def bound(path):
            inputs.check(path)
            if status['output_hashes'].get(str(path.relative_to(root))) != digest(path):
                raise ValueError('parent terminal inventory mismatch: '+str(path))
        method = manifest['config']['model']['method']; base_lr = manifest['config']['optimizer']['recipes'][method]['learning_rate']
        epochs = []; previous_update = 0
        for epoch in range(1, 6):
            file = root/'provenance'/f'epoch-{epoch:02d}.json'; bound(file)
            row = inputs.read(file); metrics = row['metrics']['summary_metrics']; update = metrics['optimizer_updates']
            ref = next(a for a in row['metrics']['array_references'] if a['key'].endswith('_global_training_losses'))
            bound(root/ref['path']); losses = verify_array_reference(root, ref)
            summary = loss_summary(losses, update-previous_update)
            if abs(summary['mean']['loss']-metrics['training_mean_loss']) > 2e-5:
                raise ValueError('loss summary differs from archived epoch')
            checkpoint = Path(row['checkpoint']['path']); bound(checkpoint); bound(checkpoint.with_suffix('.json'))
            meta = inputs.read(checkpoint.with_suffix('.json'))
            if meta['sha256'] != digest(checkpoint) or meta['identity']['source_sha'] != manifest['git_sha']:
                raise ValueError('checkpoint identity mismatch')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if payload['progress'] != dict(epoch=epoch, batch_cursor=0, update=update):
                raise ValueError('checkpoint cursor mismatch')
            prefix = 'core.clip.' if method == 'uatvr' else 'core.'
            groups = {}
            for name, value in payload['model'].items():
                if not name.startswith(prefix): continue
                key = name[len(prefix):]; tensor = inputs.tensor(value, root)
                if key not in initial or tensor.shape != initial[key].shape: raise ValueError('CLIP mapping differs')
                group = ('visual' if key.startswith('visual.') else 'text')
                g = groups.setdefault(group, dict(delta_sq=0., initial_sq=0., changed=0, tensors=0))
                g['delta_sq'] += float((tensor.double()-initial[key].double()).square().sum())
                g['initial_sq'] += float(initial[key].double().square().sum())
                g['changed'] += int(not torch.equal(tensor, initial[key])); g['tensors'] += 1
            for g in groups.values(): g['relative_l2'] = math.sqrt(g['delta_sq']/max(g['initial_sq'], 1e-30))
            temperature = inputs.tensor(payload['model'][prefix+'logit_scale'], root).exp().item()
            # Per-parameter optimizer step counters include unused parameters; report their distribution.
            steps = [v['step'] for v in payload['optimizer']['state'].values() if 'step' in v]
            counter = {str(s): steps.count(s) for s in sorted(set(steps))}
            epoch_steps = update-previous_update; total = epoch_steps*5
            used_steps = (previous_update, update-1)
            lr = [base_lr*.001*warmup_cosine(s/total, .1) for s in used_steps]
            epochs.append(dict(epoch=epoch, metrics=metrics, losses=summary, clip_parameter_change=groups,
                               clip_logit_scale=float(temperature), optimizer_step_counts=counter,
                               reconstructed_clip_lr_first_last=lr, updates_this_epoch=epoch_steps))
            previous_update = update; del payload
        log = root/'logs/worker.log'; bound(log)
        windows = re.findall(r'data_wait_s=([\d.]+) update_s=([\d.]+) global_samples_s=([\d.]+)', log.read_text())
        perf = np.asarray(windows, dtype=float)
        records.append(dict(run_id=manifest['run_id'], method=method, seed=manifest['seed'], epochs=epochs,
                            performance_window_count=len(windows), performance_window_mean=dict(zip(
                                ('data_wait_seconds', 'update_seconds', 'samples_per_second'), map(float, perf.mean(0)))) if len(windows) else None))
        print('trajectory verified', manifest['run_id'], flush=True)
    return records


def numerical(inputs, config, root, method):
    import importlib
    import numpy as np
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.full_data.runtime import settings_from_manifest
    from semantic_retrieval.full_data.qualification import initial_clip
    from semantic_retrieval.full_data.loader import CaptionDataset
    from semantic_retrieval.full_data.training import make_optimizer
    from semantic_retrieval.runtime.distributed import configure_fp32
    configure_fp32(); torch.set_num_threads(4)
    ref = config['references'][method]; source = Path(ref['path'])
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != ref['commit']:
        raise ValueError('upstream source revision changed')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream source is dirty')
    for p in source.rglob('*.py'):
        if '.git' not in p.parts: inputs.check(p)
    sys.path.insert(0, str(source))
    official = importlib.import_module('modules.modeling')
    cross_class = importlib.import_module('modules.module_cross').CrossConfig
    parent = next(p for p in config['parents'] if p['method'] == method)
    settings = settings_from_manifest(inputs.read(Path(parent['root'])/'manifest.json'))
    inputs.check(settings.clip_weights); initial = initial_clip(settings)
    torch.manual_seed(713); model = BaselineModel(method, settings.clip, initial)
    task = SimpleNamespace(max_words=32, max_frames=12, loose_type=True, linear_patch='2d',
                           sim_header='seqTransf' if method == 'uatvr' else 'meanP', cross_num_hidden_layers=4,
                           n_video_embeddings=7, n_text_embeddings=7, extra_video_cls_num=2, extra_text_cls_num=2,
                           local_rank=0, rank=0, world_size=1, n_gpu=1)
    cross = cross_class(49408, hidden_size=512, max_position_embeddings=77, num_attention_heads=8)
    cls = official.UATVR if method == 'uatvr' else official.CLIP4Clip
    torch.manual_seed(713); reference = cls(cross, dict(initial), task).float()
    mapping = model.core.state_dict() if method == 'uatvr' else {'clip.'+k:v for k,v in model.core.state_dict().items()}
    reference.load_state_dict(mapping, strict=True)
    local = {name.removeprefix('core.') if method == 'uatvr' else 'clip.'+name.removeprefix('core.'): p
             for name,p in model.named_parameters()}
    # Independently execute the pinned trainer's freeze_layer_num=0 policy.
    for name,p in reference.clip.named_parameters():
        keep=(name.startswith(('ln_final.','text_projection','logit_scale','visual.ln_post.','visual.proj'))
              or name.startswith(('transformer.resblocks.','visual.transformer.resblocks.')))
        p.requires_grad_(keep)
    for name,p in reference.named_parameters():
        if p.requires_grad != local[name].requires_grad:raise ValueError('official freeze differs: '+name)
    selected = inputs.read(config['batch'])
    for v in selected['videos'].values():
        if v['split'] != 'train': raise ValueError('audit batch is not training-only')
        inputs.check(v['path'])
    dataset = CaptionDataset(selected['videos'], selected['captions'])
    batch = tuple(torch.stack(items) for items in zip(*(dataset[i] for i in range(len(dataset)))))
    ids, masks, video, vm, identities = batch
    results = {}; array = lambda x: x.detach().cpu().numpy()
    # The rawvideo path is active in CLIP4Clip; UATVR ships this as an inactive alternative.
    np.long = np.int64; np.float = float  # Explicit NumPy 1.x compatibility aliases for upstream loader.
    loader = importlib.import_module('dataloaders.dataloader_msrvtt_retrieval')
    tokenizer = importlib.import_module('modules.tokenization_clip').SimpleTokenizer()
    raw_loader = loader.MSRVTT_DataLoader.__new__(loader.MSRVTT_DataLoader)
    raw_loader.max_words=32; raw_loader.max_frames=12; raw_loader.tokenizer=tokenizer
    raw_loader.SPECIAL_TOKEN=dict(CLS_TOKEN='<|startoftext|>', SEP_TOKEN='<|endoftext|>', MASK_TOKEN='[MASK]', UNK_TOKEN='[UNK]', PAD_TOKEN='[PAD]')
    raw_loader.features_path=str(Path(next(iter(selected['videos'].values()))['path']).parent)
    raw_loader.slice_framepos=2; raw_loader.frame_order=0
    raw_loader.rawVideoExtractor=importlib.import_module('dataloaders.rawvideo_util').RawVideoExtractor(framerate=1, size=224)
    for i, row in enumerate(selected['captions']):
        other_ids,other_mask,_,_ = raw_loader._get_text(row['official_video_id'], row['query_text'])
        results[f'input_{i}_tokens']=compare_arrays(array(ids[i]),other_ids[0],atol=0,rtol=0)
        results[f'input_{i}_text_mask']=compare_arrays(array(masks[i]),other_mask[0],atol=0,rtol=0)
        other_video,other_vm=raw_loader._get_rawvideo([row['official_video_id']])
        results[f'input_{i}_video']=compare_arrays(array(video[i]),other_video[0,:,0],atol=0,rtol=0)
        results[f'input_{i}_frame_mask']=compare_arrays(array(vm[i]),other_vm[0],atol=0,rtol=0)
    init = root/'provenance'/('gloo-'+method)
    dist.init_process_group('gloo', init_method='file://'+str(init), rank=0, world_size=1)
    try:
        def ref_forward():
            text=reference.get_sequence_output(ids,torch.zeros_like(ids),masks.long(),shaped=True)
            visual=reference.get_visual_output(video.flatten(0,1),vm.long(),shaped=True,video_frame=12)
            args=(*text,visual,masks.long(),vm.long()) if method=='uatvr' else (text,visual,masks.long(),vm.long())
            out=reference.get_similarity_logits(*args,shaped=True,loose_type=True)
            logits=out[0]; ce=(reference.loss_fct(logits)+reference.loss_fct(logits.T))/2
            mil,kl=out[1:] if method=='uatvr' and reference.training else (logits.sum()*0,logits.sum()*0)
            return dict(logits=logits,loss=ce+mil+kl,contrastive=ce,mil=mil,kl=kl),text,visual
        model.eval();reference.eval()
        with torch.no_grad():
            torch.manual_seed(991); rt, rtext, rv=ref_forward()
            torch.manual_seed(991); mt=model.encode_text(ids,masks);mv=model.encode_video(video,vm)
            results['text_pooled']=compare_arrays(array(mt['pooled']),array(rtext[0] if method=='uatvr' else rtext))
            results['valid_video_tokens']=compare_arrays(array(mv['tokens'][vm]),array(rv[vm]))
            results['eval_logits']=compare_arrays(array(model.score(mt,mv)),array(rt['logits']))
        model.train();reference.train()
        optimizer=make_optimizer(model,learning_rate=5e-5 if method=='uatvr' else 1e-4,total_steps=100)
        bert=importlib.import_module('modules.optimization').BertAdam
        groups=[]
        for decay in (.2,0.):
            for clip in (True,False):
                params=[p for n,p in reference.named_parameters() if p.requires_grad and ('clip.' in n)==clip
                        and (not any(k in n for k in ['bias','LayerNorm.bias','LayerNorm.weight']))==(decay!=0)]
                if params:groups.append(dict(params=params,weight_decay=decay,lr=(5e-5 if method=='uatvr' else 1e-4)*(.001 if clip else 1.)))
        ref_optimizer=bert(groups,lr=5e-5 if method=='uatvr' else 1e-4,warmup=.1,schedule='warmup_cosine',
                           b1=.9,b2=.98,e=1e-6,t_total=100,weight_decay=.2,max_grad_norm=1.)
        for step in range(2):  # Step zero has warmup LR=0; second step checks a real update.
            optimizer.zero_grad();ref_optimizer.zero_grad()
            torch.manual_seed(991+step); out=model(*batch)
            torch.manual_seed(991+step); expected,_,_=ref_forward()
            for key in out:results[f'step{step}_{key}']=compare_arrays(array(out[key]),array(expected[key]))
            out['loss'].backward();expected['loss'].backward()
            bad=[];maximum=0.
            for n,p in reference.named_parameters():
                other=local[n]
                if (p.grad is None)!=(other.grad is None):raise ValueError('gradient presence differs: '+n)
                if p.grad is not None:
                    comparison=compare_arrays(array(other.grad),array(p.grad));maximum=max(maximum,comparison['max_abs'])
                    if not comparison['passed']:bad.append(dict(parameter=n,**comparison))
            results[f'step{step}_gradients']=dict(passed=not bad,max_abs=maximum,failures=bad)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
            optimizer.step();ref_optimizer.step()
            with torch.no_grad():model.clip.logit_scale.clamp_(max=math.log(100));reference.clip.logit_scale.clamp_(max=math.log(100))
            bad=[];maximum=0.
            for n,p in reference.named_parameters():
                comparison=compare_arrays(array(local[n]),array(p));maximum=max(maximum,comparison['max_abs'])
                if not comparison['passed']:bad.append(dict(parameter=n,**comparison))
            results[f'step{step}_parameters']=dict(passed=not bad,max_abs=maximum,failures=bad)
        return dict(method=method,checks=results,all_passed=all(r['passed'] for r in results.values()),
                    scope='formal CLIP B/32; two real training pairs; 12 frames; CPU FP32; one Gloo rank; shared initial weights and RNG',
                    upstream_rawvideo_active=method=='clip4clip',test_access=False)
    finally:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--numeric',choices=('uatvr','clip4clip'));args=parser.parse_args()
    from scripts import manage_experiment_run as manager
    from experiment_artifacts import manifest_sha256
    from semantic_retrieval.runtime.distributed import configure_fp32
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise ValueError('audit requires hidden CUDA devices')
    root=args.run_dir.resolve();manifest=json.loads((root/'manifest.json').read_text())
    if (manifest['study']!=CONTROL.name or manifest['test_authorized'] or manifest['environment']['gpu_count']!=0
            or manifest['config']['evaluation']['audit_kind']!='official-parity-v1'):
        raise ValueError('audit scope mismatch')
    identity=argparse.Namespace(repo_root=str(ROOT),registry=str(CONTROL/'registry.jsonl'),run_dir=str(root),
                               run_id=manifest['run_id'],manifest_sha256=manifest_sha256(manifest))
    manager.verify_run(identity);inputs=Inputs(manifest);config=inputs.read(ROOT/CONTROL/'inputs/config.json')
    if args.numeric:
        result=numerical(inputs,config,root,args.numeric)
        publish(root/'provenance'/f'numeric-{args.numeric}.json',result);return
    import signal
    def deadline(_signum,_frame):raise TimeoutError('CPU audit exceeded 7200-second run budget')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(7200)
    manager.start_run(identity);started=time.monotonic()
    try:
        import shutil
        for method, reference in config['references'].items():
            directory=Path(reference['path'])
            for path in directory.rglob('*'):
                if path.is_file() and '.git' not in path.parts and (path.suffix in {'.py','.sh','.md','.gz'}):
                    inputs.check(path)
                    target=root/'provenance'/'official-sources'/method/path.relative_to(directory)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        configure_fp32();torch.set_num_threads(4)
        from semantic_retrieval.full_data.runtime import settings_from_manifest
        from semantic_retrieval.full_data.qualification import initial_clip
        settings=settings_from_manifest(inputs.read(Path(config['parents'][0]['root'])/'manifest.json'))
        inputs.check(settings.clip_weights);initial=initial_clip(settings)
        rows=trajectory(inputs,config,initial);del initial
        publish(root/'provenance/trajectories.json',dict(schema_version=1,rows=rows,
                missing_historical_measurements=['gradient norms','clip frequency','per-loss gradient contributions'],
                learning_rates='reconstructed from authenticated optimizer recipe and step counters, not historical logging'))
        for method in ('uatvr','clip4clip'):
            command=[sys.executable,__file__,'--run-dir',str(root),'--numeric',method]
            with (root/'logs'/f'numeric-{method}.log').open('xb') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        publish(root/'provenance/audit-result.json',dict(schema_version=1,elapsed_seconds=time.monotonic()-started,
                gpu_hours=0,test_access=False,training_runs=len(rows),epochs=30,
                numeric={m:json.loads((root/'provenance'/f'numeric-{m}.json').read_text())['all_passed'] for m in ('uatvr','clip4clip')}))
        manager.transition_run(argparse.Namespace(**vars(identity),expected='running',next_state='validating',exit_code=None,output=[]))
        manager.transition_run(argparse.Namespace(**vars(identity),expected='validating',next_state='completed',exit_code=0,output=[]))
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        import traceback
        publish(root/'provenance/audit-failure.json',dict(traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        state=json.loads((root/'status.json').read_text())['state']
        if state=='running':
            manager.transition_run(argparse.Namespace(**vars(identity),expected='running',next_state='validating',exit_code=None,output=[]))
            state='validating'
        if state=='validating':
            manager.transition_run(argparse.Namespace(**vars(identity),expected=state,next_state='failed',exit_code=1,output=[]))
        raise


if __name__=='__main__':main()
