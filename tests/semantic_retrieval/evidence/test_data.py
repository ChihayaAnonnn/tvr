from dataclasses import replace
from hashlib import sha256
import pytest
from test_episodes import episode_manifest
from semantic_retrieval.evidence.config import SplitBinding, InputReference
from semantic_retrieval.evidence.data import audit_component_roles, read_split_binding


def test_components_cannot_cross_match_and_value(episode_manifest):
    payload, write, _ = episode_manifest
    path = write()
    binding = SplitBinding('T_match', InputReference(path, sha256(path.read_bytes()).hexdigest()), 'reviewed_semantic')
    eps = read_split_binding(binding)
    assert eps[0].role == 'T_match'
    value = replace(eps[0], role='T_value')
    with pytest.raises(ValueError):
        audit_component_roles(eps+(value,))
    path.write_text('{}')
    with pytest.raises(ValueError):
        read_split_binding(binding)
