import pytest
from semantic_retrieval.laya_rerank.campaign import projected_hours, qualification_config
from semantic_retrieval.laya_rerank.protocol import load_fixed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]


def test_budget_counts_failed_and_live_reservations():
    assert projected_hours([{'state':'failed','elapsed_seconds':1800,'gpu_count':2},
                            {'state':'running','max_seconds':1800,'gpu_count':2}]) == 2
    with pytest.raises(ValueError): projected_hours([{'state':'failed','gpu_count':2}])


def test_qualification_is_two_rank_synthetic_only():
    q=qualification_config(load_fixed(ROOT))
    assert q['world_size']==2 and q['synthetic'] and q['random_initialization']
    assert q['max_seconds']==1800 and q['precision']=='fp32'
    assert 'parent' not in q and 'checkpoint' not in q


def test_direct_worker_without_controller_rejected(tmp_path):
    from semantic_retrieval.laya_rerank.campaign import validate_worker_launch
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_worker_launch(tmp_path,tmp_path,{},'continuous')


def test_owned_cleanup_stops_child_sessions(tmp_path):
    import os,subprocess,sys,time
    from semantic_retrieval.laya_rerank.campaign import stop_owned,owned_processes
    nonce='laya-cpu-test-'+str(os.getpid())
    child=tmp_path/'child.py';child.write_text('import time\ntime.sleep(60)\n')
    parent=tmp_path/'parent.py';parent.write_text('import subprocess,sys,time\nsubprocess.Popen([sys.executable,sys.argv[1]],start_new_session=True)\ntime.sleep(60)\n')
    process=subprocess.Popen([sys.executable,str(parent),str(child)],env={**os.environ,'LAYA_LAUNCH_NONCE':nonce},start_new_session=True)
    try:
        deadline=time.monotonic()+5
        while len(owned_processes(nonce))<2 and time.monotonic()<deadline:time.sleep(.05)
        assert len(owned_processes(nonce))==2
    finally:stop_owned(process,nonce)
    assert owned_processes(nonce)==[]


def test_worker_environment_requires_actual_rank_topology(monkeypatch):
    import platform,torch
    from semantic_retrieval.runtime.artifacts import assert_runtime_environment
    from semantic_retrieval.runtime.distributed import configure_fp32
    configure_fp32();monkeypatch.setenv('WORLD_SIZE','1')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    environment=dict(python=platform.python_version(),pytorch=str(torch.__version__),cuda=str(torch.version.cuda or 'none'),
                     gpu_model='none',gpu_count=0,precision='fp32',distributed_topology='torchrun-2')
    with pytest.raises(ValueError,match='environment'):assert_runtime_environment({'environment':environment})
    environment['distributed_topology']='single-process'
    assert assert_runtime_environment({'environment':environment})==0
