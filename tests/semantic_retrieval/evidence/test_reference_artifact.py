def test_reference_parts_externalize_large_arrays_and_verify_before_loading(config_factory,tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.models.clip import ClipBackbone
    from semantic_retrieval.evidence.model import EvidenceMatcherModel,load_reference_parts
    from semantic_retrieval.evidence.execution import _write_reference_parts
    from semantic_retrieval.data.episodes import file_sha256
    cfg=config_factory()
    clip=replace(cfg.model.base.clip,vocab_size=49408)
    weights=tmp_path/'clip.pt';torch.save(ClipBackbone(clip).state_dict(),weights)
    model=EvidenceMatcherModel(replace(cfg.model,base=replace(cfg.model.base,clip=clip,clip_weights=weights,clip_sha256=file_sha256(weights))))
    store=ArtifactArrayStore(tmp_path/'run'/'arrays')
    checkpoints=store.run_root/'checkpoints';checkpoints.mkdir()
    path=_write_reference_parts(SimpleNamespace(paths=SimpleNamespace(checkpoints=checkpoints),array_store=store),model)
    payload=torch.load(path,weights_only=True)
    assert isinstance(payload['clip']['token_embedding.weight'],dict)
    old=model.clip.token_embedding.weight.detach().clone()
    with torch.no_grad():model.clip.token_embedding.weight.zero_()
    load_reference_parts(path,model,expected_sha256=file_sha256(path))
    assert torch.equal(model.clip.token_embedding.weight,old)
    reference=payload['clip']['token_embedding.weight']['__array_reference__']
    array=store.run_root/reference['path']
    with array.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    import pytest
    with pytest.raises(ValueError):load_reference_parts(path,model,expected_sha256=file_sha256(path))
