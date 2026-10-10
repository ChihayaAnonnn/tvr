import copy
from pathlib import Path

import pytest

from semantic_retrieval.models.clip import ClipConfig


def fixture_clip():
    return ClipConfig(64, 16, 64, 1, 4, 32, 128, 64, 1, 1, fixture=True)


def test_typed_full_data_manifest_consumes_all_fields_and_refuses_mutation(tmp_path):
    from semantic_retrieval.full_data.runtime import FullDataSettings, manifest_config, settings_from_manifest
    settings = FullDataSettings('cpu_fixture', 'both', tmp_path/'clip.npz', 'a'*64, fixture_clip(), 713)
    payload = manifest_config(settings)
    assert settings_from_manifest({'config': payload}) == settings
    bad = copy.deepcopy(payload); bad['optimizer']['weight_decay'] = 0.1
    with pytest.raises(ValueError, match='config'):
        settings_from_manifest({'config': bad})
    bad = copy.deepcopy(payload); bad['model']['unconsumed'] = True
    with pytest.raises(ValueError, match='config'):
        settings_from_manifest({'config': bad})
    with pytest.raises(ValueError):
        FullDataSettings('baseline', 'clip4clip', tmp_path/'clip.npz', 'a'*64, fixture_clip(), 713)
    with pytest.raises(ValueError):
        FullDataSettings('cpu_fixture', 'both', Path('relative.npz'), 'a'*64, fixture_clip(), 713)


def test_four_rank_qualification_settings_are_formal_and_both_methods(tmp_path):
    from semantic_retrieval.full_data.runtime import FullDataSettings, manifest_config
    clip = ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)
    cfg = FullDataSettings('gpu_qualification', 'both', tmp_path/'random.npz', 'a'*64, clip, 713)
    manifest = manifest_config(cfg)
    assert manifest['model']['temporal_layers'] == 4
    assert manifest['evaluation']['frames'] == 12 and manifest['evaluation']['text_length'] == 32
    assert {k: v['global_batch'] for k, v in manifest['optimizer']['recipes'].items()} == {'uatvr': 64, 'clip4clip': 128}
    with pytest.raises(ValueError):
        FullDataSettings('gpu_qualification', 'clip4clip', tmp_path/'random.npz', 'a'*64, clip, 713)


def test_resume_is_explicit_bound_and_never_allowed_for_software_fixture(tmp_path):
    from semantic_retrieval.full_data.runtime import ResumeReference, FullDataSettings, manifest_config, settings_from_manifest
    reference = ResumeReference('parent-run', tmp_path/'checkpoint.pt', 'b'*64, 'c'*64)
    with pytest.raises(ValueError):
        FullDataSettings('cpu_fixture', 'both', tmp_path/'clip.npz', 'a'*64, fixture_clip(), 713, resume=reference)
    clip = ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)
    cfg = FullDataSettings('baseline', 'uatvr', tmp_path/'clip.pt', 'a'*64, clip, 713,
                           tmp_path/'source.json', 'd'*64, reference)
    assert settings_from_manifest({'config': manifest_config(cfg)}) == cfg


def test_two_rank_topology_is_sealed_with_same_global_batches(tmp_path):
    from semantic_retrieval.full_data.runtime import FullDataSettings, manifest_config, settings_from_manifest
    from semantic_retrieval.full_data.campaign import _spec
    clip = ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)
    cfg = FullDataSettings('gpu_qualification', 'both', tmp_path/'random.npz', 'a'*64, clip, 713, world_size=2)
    packet = _spec(cfg, ['0', '3'], [], [])
    assert packet['environment']['gpu_count'] == 2
    assert packet['environment']['distributed_topology'] == 'torchrun-2'
    assert '--nproc_per_node=2' in packet['command']
    assert cfg.recipe('uatvr')['global_batch'] == 64 and cfg.recipe('clip4clip')['global_batch'] == 128
    assert settings_from_manifest(packet) == cfg
    bad = copy.deepcopy(packet); bad['config']['evaluation']['world_size'] = 3
    with pytest.raises(ValueError): settings_from_manifest(bad)
    with pytest.raises(ValueError): _spec(cfg, ['0', '0'], [], [])
