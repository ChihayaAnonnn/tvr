import logging
import math
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import main_task_retrieval
import modules.modeling as modeling
from main_task_retrieval import train_epoch
from modules.modeling import UATVR
from modules.module_cross import Transformer as TransformerClip
from modules.stochastic_prototype_ranking import RSPRCore
from modules.until_module import KLdivergence, MILNCELoss_BoF, MultiPositiveCrossEn
from prob_models.pie_model import PIENet
from prob_models.uncertainty_module import UncertaintyModuleImage


def _rspr_config(mode, **overrides):
    values = {
        "rspr_mode": mode,
        "rspr_sample_count": 4,
        "rspr_eval_sample_count": 8,
        "rspr_match_temperature": 0.2,
        "rspr_rank_temperature": 0.3,
        "rspr_hard_negatives": 2,
        "rspr_prior_std": 0.4,
        "rspr_match_mode": "hard",
        "rspr_eval_seed": 17,
        "rspr_detach_samples": False,
        "rspr_prob_temperature": 0.07,
        "rspr_prob_weight": 0.1,
        "rspr_rank_weight": 0.1,
        "rspr_anchor_weight": 1e-4,
        "n_video_embeddings": 7,
        "n_text_embeddings": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_model(mode, **overrides):
    model = UATVR.__new__(UATVR)
    nn.Module.__init__(model)
    model.task_config = _rspr_config(mode, **overrides)
    return model


@pytest.mark.parametrize("mode", ("mean", "stochastic"))
def test_new_rspr_modes_initialize_only_rspr_probability_path(mode):
    model = _bare_model(mode)

    model._initialize_probability_path(embed_dim=4)

    assert model.rspr_mode == mode
    assert isinstance(model.multi_positive_loss, MultiPositiveCrossEn)
    assert isinstance(model.rspr, RSPRCore)
    assert model.rspr.sample_count == 4
    assert model.rspr.eval_sample_count == 8
    assert model.rspr.matcher.temperature == pytest.approx(0.2)
    assert model.rspr.matcher.hard_max is True
    assert model.rspr.rank_loss.temperature == pytest.approx(0.3)
    assert model.rspr.rank_loss.hard_negative_count == 2
    assert not hasattr(model, "pie_net_video")
    assert not hasattr(model, "uncertain_net_video")
    assert not hasattr(model, "loss_MIL_fct")
    assert not hasattr(model, "vib_loss")


def test_final_model_initialization_preserves_rspr_specialized_output_heads():
    model = _bare_model("stochastic")
    model.config = SimpleNamespace(initializer_range=0.02)
    model.rspr_mode = "stochastic"
    model.multi_positive_loss = MultiPositiveCrossEn()
    model.rspr = None
    model.global_init_probe = nn.Linear(4, 4)

    model._initialize_model_weights(embed_dim=4)

    expected_logvar = math.log(model.task_config.rspr_prior_std**2)
    for distribution in (
        model.rspr.text_distribution,
        model.rspr.video_distribution,
    ):
        torch.testing.assert_close(
            distribution.mean_head[-1].weight,
            torch.zeros_like(distribution.mean_head[-1].weight),
        )
        torch.testing.assert_close(
            distribution.mean_head[-1].bias,
            torch.zeros_like(distribution.mean_head[-1].bias),
        )
        torch.testing.assert_close(
            distribution.logvar_head[-1].weight,
            torch.zeros_like(distribution.logvar_head[-1].weight),
        )
        torch.testing.assert_close(
            distribution.logvar_head[-1].bias,
            torch.full_like(distribution.logvar_head[-1].bias, expected_logvar),
        )


def test_legacy_initializes_only_original_probability_path():
    model = _bare_model("legacy")

    model._initialize_probability_path(embed_dim=4)

    assert model.rspr is None
    assert isinstance(model.pie_net_video, PIENet)
    assert isinstance(model.uncertain_net_video, UncertaintyModuleImage)
    assert isinstance(model.pie_net_text, PIENet)
    assert isinstance(model.uncertain_net_text, UncertaintyModuleImage)
    assert isinstance(model.loss_MIL_fct, MILNCELoss_BoF)
    assert isinstance(model.vib_loss, KLdivergence)


def test_off_initializes_no_probability_path():
    model = _bare_model("off")

    model._initialize_probability_path(embed_dim=4)

    assert model.rspr is None
    assert isinstance(model.multi_positive_loss, MultiPositiveCrossEn)
    assert not hasattr(model, "pie_net_video")
    assert not hasattr(model, "uncertain_net_video")
    assert not hasattr(model, "loss_MIL_fct")
    assert not hasattr(model, "vib_loss")


def test_off_similarity_does_not_call_legacy_probability_branch(monkeypatch):
    model = _bare_model("off")
    model._initialize_probability_path(embed_dim=4)
    model.sim_header = "meanP"
    model.clip = nn.Module()
    model.clip.logit_scale = nn.Parameter(torch.tensor(0.0))
    model.text_weight_fc = nn.Linear(4, 1)
    model.video_weight_fc = nn.Linear(4, 1)
    model.train()

    legacy_calls = []

    def forbidden_legacy_call(self, *args, **kwargs):
        del self, args, kwargs
        legacy_calls.append(True)
        raise AssertionError("legacy probability branch was called")

    model.probabilistic_video = MethodType(forbidden_legacy_call, model)
    model.probabilistic_text = MethodType(forbidden_legacy_call, model)
    monkeypatch.setattr(modeling, "allgather", lambda tensor, _config: tensor)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    sequence_output = torch.randn(2, 1, 4)
    text_token = torch.randn(2, 3, 4)
    visual_output = torch.randn(2, 2, 4)
    attention_mask = torch.ones(2, 3, dtype=torch.long)
    video_mask = torch.ones(2, 2, dtype=torch.long)

    logits, legacy_mil, legacy_kl = model._loose_similarity(
        sequence_output,
        text_token,
        visual_output,
        attention_mask,
        video_mask,
        sim_header="meanP",
    )

    assert logits.shape == (2, 2)
    torch.testing.assert_close(legacy_mil, torch.zeros_like(legacy_mil))
    torch.testing.assert_close(legacy_kl, torch.zeros_like(legacy_kl))
    assert legacy_calls == []


@pytest.mark.filterwarnings("error:To copy construct from a tensor")
def test_wti_mask_conversion_is_warning_free_and_numerically_unchanged():
    model = _bare_model("off")
    model.text_weight_fc = nn.Linear(2, 1, bias=False)
    model.video_weight_fc = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.text_weight_fc.weight.copy_(torch.tensor([[0.5, -0.25]]))
        model.video_weight_fc.weight.copy_(torch.tensor([[0.2, 0.4]]))
    text_token = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], [[1.0, 1.0], [-1.0, 2.0], [0.0, 3.0]]]
    )
    video_token = torch.tensor(
        [[[1.0, 1.0], [2.0, 0.0]], [[0.0, 1.0], [1.0, -1.0]]]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 1]])
    video_mask = torch.tensor([[1, 0], [1, 1]])

    logits = model.weighted_token_wise_intersection(
        text_token, video_token, attention_mask, video_mask
    )

    expected = torch.tensor(
        [[1.0, 1.0], [2.6344707012, 1.7374258041]]
    )
    torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-6)


def _refinement_harness():
    torch.manual_seed(13)
    model = _bare_model("off")
    model.frame_position_embeddings = nn.Embedding(8, 4)
    model.word_position_embeddings = nn.Embedding(8, 4)
    model.transformerClip = TransformerClip(width=4, layers=1, heads=1)
    model.extra_cls_frame_num = 2
    model.extra_cls_text_num = 1
    model.eval()
    return model


def test_refine_video_tokens_matches_pre_refactor_fixture():
    model = _refinement_harness()
    visual_output = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    video_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    expected_tokens = torch.tensor(
        [
            [
                [0.5824429989, 1.1145597696, 1.4509327412, 0.3156767190],
                [0.8761370182, -0.2908610106, 0.4443084598, 1.1144429445],
                [1.4343609810, 3.2586917877, 2.4575881958, 2.0383012295],
                [-0.3597857654, 0.3958700597, -0.2959058881, 1.8361248970],
                [0.2991784215, -0.3187335730, -1.2844080925, 0.1201014817],
            ],
            [
                [3.0987992287, 3.3416786194, 3.9227719307, 2.8559689522],
                [3.3035588264, 2.0199337006, 2.8638439178, 3.5913043022],
                [3.9532661438, 5.5449275970, 4.8760890961, 4.5265512466],
                [-0.3319370747, 0.3358966112, -0.2762191296, 1.8808038235],
                [0.3082810640, -0.3896616995, -1.2758073807, 0.1822590232],
            ],
        ]
    )
    expected_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    )

    refined, refined_mask = model._refine_video_tokens(visual_output, video_mask)

    torch.testing.assert_close(refined, expected_tokens, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(refined_mask, expected_mask, rtol=1e-6, atol=1e-6)


def test_refine_text_tokens_matches_pre_refactor_fixture():
    model = _refinement_harness()
    text_token = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 7
    attention_mask = torch.tensor([[1, 0], [1, 1]])
    expected_tokens = torch.tensor(
        [
            [
                [0.8837332726, 2.5820684433, 1.2026629448, 0.4305770993],
                [1.2679858208, 2.5436396599, 2.6260411739, 1.6302759647],
                [-0.9538241625, 1.0309655666, -2.4662959576, 0.8192622066],
            ],
            [
                [3.1773533821, 4.7428903580, 3.4758789539, 2.6839699745],
                [3.6166582108, 4.6564612389, 4.9263939857, 3.9129254818],
                [-1.0078850985, 0.9850642681, -2.5317373276, 0.7582129240],
            ],
        ]
    )
    expected_mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])

    refined, refined_mask = model._refine_text_tokens(text_token, attention_mask)

    torch.testing.assert_close(refined, expected_tokens, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(refined_mask, expected_mask, rtol=1e-6, atol=1e-6)


def test_assemble_training_loss_uses_exact_unweighted_components_and_scales():
    model = _bare_model(
        "stochastic",
        rspr_prob_weight=0.2,
        rspr_rank_weight=0.3,
        rspr_anchor_weight=0.4,
    )
    dsa_loss = torch.tensor(1.25, requires_grad=True)
    probability_loss = torch.tensor(2.5, requires_grad=True)
    rank_loss = torch.tensor(3.75, requires_grad=True)
    rspr_output = SimpleNamespace(
        anchor_kl=torch.tensor(4.5, requires_grad=True),
        pair_uncertainty=torch.tensor([[0.2, 0.4]]),
        text_distribution=SimpleNamespace(logvar=torch.log(torch.tensor([[2.0, 4.0]]))),
        video_distribution=SimpleNamespace(logvar=torch.log(torch.tensor([[3.0, 5.0]]))),
    )

    loss = model._assemble_training_loss(
        dsa_loss,
        probability_loss,
        rank_loss,
        rspr_output,
        rspr_rank_scale=0.5,
        rspr_anchor_scale=0.25,
    )

    expected = 1.25 + 0.2 * 2.5 + 0.3 * 0.5 * 3.75 + 0.4 * 0.25 * 4.5
    torch.testing.assert_close(loss, torch.tensor(expected))
    expected_diagnostics = {
        "dsa": 1.25,
        "prob": 2.5,
        "rank": 3.75,
        "anchor": 4.5,
        "pair_uncertainty_mean": 0.3,
        "text_variance_mean": 3.0,
        "video_variance_mean": 4.0,
    }
    assert set(model.last_loss_diagnostics) == set(expected_diagnostics)
    for name, expected_value in expected_diagnostics.items():
        assert model.last_loss_diagnostics[name].requires_grad is False
        torch.testing.assert_close(
            model.last_loss_diagnostics[name], torch.tensor(expected_value)
        )


class _RecordingRSPRCore(RSPRCore):
    def forward(self, text_tokens, text_mask, video_tokens, video_mask, **kwargs):
        self.last_inputs = (text_tokens, text_mask, video_tokens, video_mask, kwargs)
        return super().forward(
            text_tokens, text_mask, video_tokens, video_mask, **kwargs
        )


class _RecordingRankLoss(nn.Module):
    def __init__(self, rank_loss):
        super().__init__()
        self.rank_loss = rank_loss

    def bidirectional(self, stochastic_scores, group_ids, mining_logits):
        self.last_mining_logits = mining_logits
        return self.rank_loss.bidirectional(
            stochastic_scores, group_ids, mining_logits
        )


def _forward_rspr_harness(mode):
    sample_count = 1 if mode == "mean" else 4
    model = _bare_model(mode, rspr_sample_count=sample_count)
    model.rspr_mode = mode
    model.loose_type = True
    model.loss_fct = modeling.CrossEn()
    model.multi_positive_loss = MultiPositiveCrossEn()
    model.rspr = _RecordingRSPRCore(
        dim=4,
        sample_count=sample_count,
        eval_sample_count=8,
        match_temperature=0.2,
        rank_temperature=0.3,
        hard_negative_count=2,
        prior_std=0.4,
        hard_max=False,
        eval_seed=17,
    )
    model.rspr.rank_loss = _RecordingRankLoss(model.rspr.rank_loss)
    model.refined_text = nn.Parameter(torch.randn(3, 4, 4))
    model.refined_video = nn.Parameter(torch.randn(3, 3, 4))

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped):
        del token_type_ids, attention_mask, shaped
        return torch.ones(input_ids.size(0), 1, 4), self.refined_text[:, :3]

    def get_visual_output(self, video, video_mask, shaped, video_frame):
        del video, video_mask, shaped, video_frame
        return self.refined_video[:, :2]

    def get_similarity_logits(self, *args, **kwargs):
        del args, kwargs
        raw_wti_logits = self.refined_text[:, 0] @ self.refined_video[:, 0].T
        retrieve_logits = 3.0 * raw_wti_logits
        attention_mask = torch.tensor(
            [[1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )
        video_mask = torch.ones(3, 3, dtype=torch.long)
        return (
            retrieve_logits,
            retrieve_logits.new_tensor(50.0),
            retrieve_logits.new_tensor(70.0),
            self.refined_text,
            attention_mask,
            self.refined_video,
            video_mask,
            3,
            2,
            raw_wti_logits,
        )

    model.get_sequence_output = MethodType(get_sequence_output, model)
    model.get_visual_output = MethodType(get_visual_output, model)
    model.get_similarity_logits = MethodType(get_similarity_logits, model)
    model.train()
    return model


def _forward_inputs():
    return (
        torch.zeros(3, 1, 3, dtype=torch.long),
        torch.zeros(3, 1, 3, dtype=torch.long),
        torch.ones(3, 1, 3, dtype=torch.long),
        torch.zeros(3, 1, 1, 1, 1, 1, 1),
        torch.ones(3, 1, 2, dtype=torch.long),
    )


@pytest.mark.parametrize("mode", ("mean", "stochastic"))
def test_forward_rspr_modes_use_core_terms_without_legacy_losses(monkeypatch, mode):
    torch.manual_seed(29)
    model = _forward_rspr_harness(mode)
    gather_calls = []

    def identity_gather(ids, _config):
        gather_calls.append(ids)
        return ids

    monkeypatch.setattr(modeling, "allgather", identity_gather)
    group_ids = torch.tensor([0, 1, 2])

    loss = model(
        *_forward_inputs(),
        group_ids=group_ids,
        rspr_rank_scale=0.6,
        rspr_anchor_scale=0.4,
    )

    diagnostics = model.last_loss_diagnostics
    expected = (
        diagnostics["dsa"]
        + model.task_config.rspr_prob_weight * diagnostics["prob"]
        + model.task_config.rspr_rank_weight * 0.6 * diagnostics["rank"]
        + model.task_config.rspr_anchor_weight * 0.4 * diagnostics["anchor"]
    )
    torch.testing.assert_close(loss.detach(), expected)
    assert float(loss.detach()) < float((diagnostics["dsa"] + 50.0 + 70.0))
    assert len(gather_calls) == 1
    text_tokens, text_mask, video_tokens, video_mask, call_kwargs = (
        model.rspr.last_inputs
    )
    assert text_tokens.shape == (3, 3, 4)
    assert text_mask.shape == (3, 3)
    assert video_tokens.shape == (3, 2, 4)
    assert video_mask.shape == (3, 2)
    assert call_kwargs["sample_count"] == (1 if mode == "mean" else 4)
    assert call_kwargs["mean_only"] is (mode == "mean")
    if mode == "mean":
        torch.testing.assert_close(diagnostics["rank"], torch.tensor(0.0))
    else:
        assert diagnostics["rank"] > 0
        raw_wti_logits = model.refined_text[:, 0] @ model.refined_video[:, 0].T
        torch.testing.assert_close(
            model.rspr.rank_loss.last_mining_logits, raw_wti_logits.detach()
        )
        assert model.rspr.rank_loss.last_mining_logits.requires_grad is False

    loss.backward()
    rspr_gradients = [
        parameter.grad
        for parameter in model.rspr.parameters()
        if parameter.requires_grad
    ]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in rspr_gradients
    )


def _gradient_core_and_output():
    torch.manual_seed(41)
    core = RSPRCore(
        dim=4,
        sample_count=4,
        eval_sample_count=8,
        match_temperature=0.2,
        rank_temperature=0.3,
        hard_negative_count=2,
        prior_std=0.4,
    )
    text_tokens = torch.randn(3, 3, 4)
    video_tokens = torch.randn(3, 2, 4)
    output = core(
        text_tokens,
        torch.ones(3, 3, dtype=torch.long),
        video_tokens,
        torch.ones(3, 2, dtype=torch.long),
        sample_count=4,
    )
    return core, output


def _assert_distribution_output_head_gradients(core):
    for distribution in (
        core.text_distribution,
        core.video_distribution,
    ):
        for output_head in (distribution.mean_head, distribution.logvar_head):
            gradient = output_head[-1].weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0


def test_probability_only_loss_reaches_both_distribution_heads():
    core, output = _gradient_core_and_output()
    group_ids = torch.tensor([0, 1, 2])

    probability_loss, _ = MultiPositiveCrossEn().bidirectional(
        output.probabilistic_logits / 0.07, group_ids
    )
    probability_loss.backward()

    _assert_distribution_output_head_gradients(core)


def test_rank_only_loss_reaches_distribution_heads_but_not_mining_logits():
    core, output = _gradient_core_and_output()
    group_ids = torch.tensor([0, 1, 2])
    mining_logits = torch.randn(3, 3, requires_grad=True)

    rank_loss, _, _ = core.rank_loss.bidirectional(
        output.stochastic_pair_scores, group_ids, mining_logits
    )
    rank_loss.backward()

    _assert_distribution_output_head_gradients(core)
    assert mining_logits.grad is None


@pytest.mark.parametrize(("mode", "expected_aux"), (("legacy", 120.0), ("off", 0.0)))
def test_forward_legacy_and_off_keep_objectives_isolated(monkeypatch, mode, expected_aux):
    model = _forward_rspr_harness("mean")
    model.rspr_mode = mode
    model.task_config.rspr_mode = mode
    model.rspr = None
    monkeypatch.setattr(modeling, "allgather", lambda ids, _config: ids)

    loss = model(*_forward_inputs(), group_ids=torch.tensor([0, 1, 2]))

    wti_logits = 3.0 * (model.refined_text[:, 0] @ model.refined_video[:, 0].T)
    dsa_loss, _ = MultiPositiveCrossEn().bidirectional(
        wti_logits, torch.tensor([0, 1, 2])
    )
    torch.testing.assert_close(loss, dsa_loss + expected_aux)


def _freeze_harness():
    model = _bare_model("stochastic", rspr_sample_count=4)
    model.clip = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    model.transformerClip = nn.Linear(4, 4)
    model.frame_position_embeddings = nn.Embedding(8, 4)
    model.word_position_embeddings = nn.Embedding(8, 4)
    model.text_weight_fc = nn.Linear(4, 1)
    model.video_weight_fc = nn.Linear(4, 1)
    model.rspr = RSPRCore(
        dim=4,
        sample_count=4,
        eval_sample_count=8,
        match_temperature=0.2,
        rank_temperature=0.3,
        hard_negative_count=2,
        prior_std=0.4,
    )
    return model


def test_freeze_clip_leaves_dsa_and_rspr_trainable():
    model = _freeze_harness()
    args = SimpleNamespace(
        rspr_mode="stochastic", rspr_freeze_clip=True, rspr_freeze_dsa=False
    )

    main_task_retrieval.apply_rspr_freeze_contract(model, args)

    assert not any(parameter.requires_grad for parameter in model.clip.parameters())
    assert all(parameter.requires_grad for parameter in model.transformerClip.parameters())
    assert all(parameter.requires_grad for parameter in model.text_weight_fc.parameters())
    assert all(parameter.requires_grad for parameter in model.rspr.parameters())


def test_freeze_clip_and_dsa_leave_only_rspr_in_optimizer():
    model = _freeze_harness()
    args = SimpleNamespace(
        rspr_mode="stochastic", rspr_freeze_clip=True, rspr_freeze_dsa=True
    )

    main_task_retrieval.apply_rspr_freeze_contract(model, args)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
    )

    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        if group["params"]
        for parameter in group["params"]
    }
    assert optimized_parameters
    assert optimized_parameters == {id(parameter) for parameter in model.rspr.parameters()}


def test_freeze_dsa_supports_mean_pooling_model_shape():
    model = _freeze_harness()
    del model.transformerClip
    del model.frame_position_embeddings
    del model.word_position_embeddings
    args = SimpleNamespace(
        rspr_mode="mean", rspr_freeze_clip=False, rspr_freeze_dsa=True
    )

    main_task_retrieval.apply_rspr_freeze_contract(model, args)

    assert not any(
        parameter.requires_grad for parameter in model.text_weight_fc.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.video_weight_fc.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.clip.parameters())
    assert all(parameter.requires_grad for parameter in model.rspr.parameters())


class _DiagnosticTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        del args, kwargs
        self.last_loss_diagnostics = {
            "dsa": torch.tensor(1.0),
            "prob": torch.tensor(2.0),
            "rank": torch.tensor(3.0),
            "anchor": torch.tensor(4.0),
            "pair_uncertainty_mean": torch.tensor(5.0),
            "text_variance_mean": torch.tensor(6.0),
            "video_variance_mean": torch.tensor(7.0),
        }
        return self.weight.square()


def test_train_log_appends_unweighted_rspr_diagnostics(monkeypatch, caplog):
    logger = logging.getLogger("test.rspr.train.logging")
    monkeypatch.setattr(main_task_retrieval, "logger", logger, raising=False)
    caplog.set_level(logging.INFO, logger=logger.name)
    model = _DiagnosticTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = tuple(torch.zeros(2, dtype=torch.long) for _ in range(5))
    args = SimpleNamespace(
        n_display=1,
        gradient_accumulation_steps=1,
        epochs=1,
        rspr_warmup_epochs=1.0,
    )

    train_epoch(
        0,
        args,
        model,
        [batch],
        torch.device("cpu"),
        1,
        optimizer,
        None,
        0,
    )

    step_logs = [
        record.getMessage()
        for record in caplog.records
        if "LR clip=" in record.getMessage()
    ]
    assert len(step_logs) == 1
    for fragment in (
        "dsa=1.0000",
        "prob=2.0000",
        "rank=3.0000",
        "anchor=4.0000",
        "text_var=6.0000",
        "video_var=7.0000",
    ):
        assert fragment in step_logs[0]
