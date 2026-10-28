"""Read-only historical phase certification before any scientific aggregation."""
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from experiment_artifacts import manifest_sha256,parse_registry_text,strict_json_loads
from scripts import manage_experiment_run as manager
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.input_bindings import manifest_input_references


def require(condition):
    if not condition:raise ValueError('scientific analysis input invariant failed')


def committed(repo,path):
    safe_path(path)
    raw=subprocess.check_output(['git','show','HEAD:'+path.relative_to(repo).as_posix()],cwd=repo)
    if raw!=path.read_bytes():raise ValueError('analysis control differs from committed source')
    return raw


@dataclass(frozen=True)
class CertifiedRun:
    repo: Path
    root: Path
    manifest: dict
    status: dict
    spec: dict

    def read_json(self,relative):
        if manager._safe_file_sha256(self.root,relative)!=self.status['output_hashes'].get(relative):
            raise ValueError('consumed output differs from committed terminal inventory')
        return manager._read_json(self.root/relative,'certified output')

    def config(self):
        from semantic_retrieval.evidence.config import load_evidence_config,evidence_manifest_config
        self.read_json('provenance/worker-config.json')
        cfg=load_evidence_config(self.root/'provenance/worker-config.json');rt=cfg.runtime
        expected=(str(self.root),self.manifest['run_id'],manifest_sha256(self.manifest),self.manifest['git_sha'])
        registry=self.repo/self.manifest['registry_path']
        if rt is None or rt.registry_path!=registry or rt.study_path!=registry.with_name('study.yaml') or (str(rt.run_dir),rt.run_id,rt.manifest_sha256,rt.source_sha)!=expected or evidence_manifest_config(cfg,eval_split=rt.eval_split)!=self.manifest['config']:
            raise ValueError('sealed effective configuration identity mismatch')
        from semantic_retrieval.evidence.runtime import validate_evidence_binding
        validate_evidence_binding(self.repo,self.manifest,cfg,rt)
        return cfg



def certify_phase(repo,control,phase_path,summary_path,artifact_root):
    """Certify the entire completed phase; never publish, reserve, repair or launch."""
    repo=Path(repo);control=Path(control)
    phase=manager._load_committed_phase(repo,Path(phase_path))
    summary=strict_json_loads(committed(repo,repo/summary_path).decode())
    records=parse_registry_text(committed(repo,control/'registry.jsonl').decode())
    if summary['phase']!=phase['phase'] or summary['phase_path']!=str(phase_path) or summary['study']!=phase['study'] or len(summary['runs'])!=len(phase['runs']):
        raise ValueError('committed phase summary identity mismatch')
    # Check the complete terminal cohort before inspecting any scientific outputs.
    reservations=[manager._reservation_for_spec(s,records,Path(artifact_root)) for s in phase['runs']]
    for reservation,row in zip(reservations,summary['runs']):
        terminal=[r for r in records if r['run_id']==reservation['run_id'] and r['event'] in manager.TERMINAL_STATES]
        if len(terminal)!=1 or terminal[0]['state']!='completed' or row['state']!='completed' or row['run_id']!=reservation['run_id'] or row['manifest_sha256']!=reservation['manifest_sha256'] or terminal[0]['manifest_sha256']!=reservation['manifest_sha256']:
            raise ValueError('analysis requires the entire committed completed phase')
    result=[]
    for spec,reservation,row in zip(phase['runs'],reservations,summary['runs']):
        root=Path(reservation['artifact_path']);manifest,status=manager._load_run(root)
        if status['state']!='completed' or manifest_sha256(manifest)!=reservation['manifest_sha256'] or manifest['study']!=phase['study'] or manifest['registry_path']!=(control/'registry.jsonl').relative_to(repo).as_posix():
            raise ValueError('manifest/reservation/lifecycle identity mismatch')
        command=manager._render_command(spec['command'],run_id=manifest['run_id'],run_dir=root)
        if hashlib.sha256(command).hexdigest()!=manifest['command_sha256']:raise ValueError('command differs from sealed phase specification')
        if any(manifest[k]!=spec[k] for k in manager.RUN_SPEC_FIELDS) or manifest['tracked_clean'] is not True or manifest['source_patch_sha256'] is not None:
            raise ValueError('manifest differs from committed phase specification')
        subprocess.run(['git','merge-base','--is-ancestor',manifest['git_sha'],'HEAD'],cwd=repo,check=True,capture_output=True)
        source_phase=subprocess.check_output(['git','show',manifest['git_sha']+':'+str(phase_path)],cwd=repo)
        if source_phase!=(repo/phase_path).read_bytes():raise ValueError('phase differs from sealed source snapshot')
        if 'output_hashes_reference' in row:
            reference=row['output_hashes_reference']
            safe_path(root/'status.json')
            if reference!=dict(path=str(root/'status.json'),sha256=file_sha256(root/'status.json'),json_pointer='/output_hashes',entry_count=len(status['output_hashes'])):
                raise ValueError('committed terminal status reference mismatch')
        elif row.get('output_hashes')!=status['output_hashes']:
            raise ValueError('committed terminal inventory mismatch')
        if manager._all_output_hashes(root)!=status['output_hashes']:
            raise ValueError('actual terminal output inventory mismatch')
        for name,sha in manifest_input_references(repo,manifest).items():
            path=Path(name);path=path if path.is_absolute() else repo/path;safe_path(path)
            if file_sha256(path)!=sha:raise ValueError('sealed external input hash mismatch')
        result.append(CertifiedRun(repo,root,manifest,status,spec))
    return tuple(result)
