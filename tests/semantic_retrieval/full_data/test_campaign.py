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
