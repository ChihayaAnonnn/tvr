import hashlib

import pytest
import torch


def components():
    from semantic_retrieval.models.clip import ClipBackbone, ClipConfig

    return ClipBackbone(ClipConfig(8, 8, 64, 1, 4, 5, 20, 8, 2, 1))


def test_encoding_matches_retained_clip_calculation():
    model = components().eval()
    ids = torch.tensor([[18, 2, 19, 0, 0]])
    mask = ids != 0
    tokens, pooled = model.encode_text(ids, mask)
    hidden = model.token_embedding(ids) + model.positional_embedding
    hidden = model.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
    reference = model.ln_final(hidden) @ model.text_projection
    torch.testing.assert_close(tokens[mask], reference[mask])
    torch.testing.assert_close(pooled, reference[:, 2])
    video = torch.randn(2, 3, 3, 8, 8)
    fm = torch.tensor([[True, True, False], [True, False, False]])
    frames = model.encode_frames(video, fm)
    assert frames.shape == (2, 3, 8) and frames.dtype == torch.float32
    assert torch.isfinite(frames).all() and (frames[~fm] == 0).all()
    ref = model.visual.ln_post(model.visual(video[fm])[:, 0]) @ model.visual.proj
    torch.testing.assert_close(frames[fm], ref)


@pytest.mark.parametrize("defect", ["none", "missing", "extra", "shape", "hash"])
def test_local_weight_integrity(tmp_path, defect):
    from semantic_retrieval.models.weights import load_clip_state

    model = components()
    state = model.state_dict()
    key = next(iter(state))
    if defect == "missing":
        state.pop(key)
    elif defect == "extra":
        state["legacy.foo"] = torch.zeros(1)
    elif defect == "shape":
        state[key] = torch.zeros(7)
    path = tmp_path / "clip.pt"
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if defect == "hash":
        digest = "0" * 64
    if defect == "none":
        load_clip_state(path, digest, model)
    else:
        with pytest.raises(ValueError):
            load_clip_state(path, digest, model)


def test_tokenizer_eos_padding_and_truncation():
    from semantic_retrieval.data.tokenizer import ClipTokenizer

    tokenizer = ClipTokenizer()
    ids, mask = tokenizer.encode_batch(["", "a long sentence with many words"], 4)
    assert ids.shape == mask.shape == (2, 4)
    assert mask.dtype == torch.bool
    assert ids[0].tolist() == [49406, 49407, 0, 0]
    assert ids[1, -1] == 49407 and mask[1].all()
    with pytest.raises(ValueError):
        tokenizer.encode_batch(["a"], 1)
