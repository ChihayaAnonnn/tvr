from pathlib import Path
import pytest


def test_scientific_inputs_bind_full_certificate_and_all_video_indices_without_mixed_annotation():
    from semantic_retrieval.full_data.campaign import scientific_inputs
    repo = Path(__file__).resolve().parents[3]
    refs, certificate = scientific_inputs(repo)
    names = [r['path'] for r in refs]
    assert len([n for n in names if 'D0-9000-videos-' in n]) == 10
    assert str(certificate) in names and not any('MSRVTT_v2.json' in n or 'test' in Path(n).name for n in names)
    assert any(n.endswith('.input-index.json') and 'complete-cohort' in n for n in names)


def test_torchrun_auxiliary_outputs_are_bound_to_the_reserved_run():
    from semantic_retrieval.full_data.campaign import _spec, RANDOM, FORMAL
    from semantic_retrieval.full_data.runtime import FullDataSettings
    packet = _spec(FullDataSettings('gpu_qualification', 'both', RANDOM, 'a'*64, FORMAL, 713), ['0', '1', '2', '3'], [], [])
    argv = packet['command']; index = argv.index('--log-dir')
    assert argv[index+1] == '{artifact_run_dir}/logs/elastic'
    assert index < argv.index('scripts/qualify_full_data_retrieval.py')
