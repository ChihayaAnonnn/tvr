import numpy as np

from prob_models.retrieval_risk_metrics import (
    build_eval_positive_mask,
    evaluate_intrinsic_uncertainty,
    summarize_csd_components,
    summarize_logvar,
)


def test_intrinsic_metrics_use_one_value_per_directional_query():
    scores = np.array(
        [[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [5.0, 3.0]],
        dtype=np.float32,
    )
    mask = build_eval_positive_mask(4, 2, cut_off_points=[2, 4])

    t2v = evaluate_intrinsic_uncertainty(
        scores, mask, np.array([0.1, 0.2, 0.3, 0.9]), 20, 0
    )
    v2t = evaluate_intrinsic_uncertainty(
        scores.T, mask.T, np.array([0.1, 0.9]), 20, 1
    )

    assert t2v["num_queries"] == 4
    assert v2t["num_queries"] == 2


def test_logvar_summary_reports_clamp_rates():
    summary = summarize_logvar(
        np.array([[-12.0, -8.0], [4.0, -7.0]]), -12.0, 4.0
    )

    assert summary["clamp_low_fraction"] == 0.25
    assert summary["clamp_high_fraction"] == 0.25
    assert summary["variance"]["mean"] > 0


def test_chunked_csd_summary_matches_direct_components():
    text_mu = np.array([[1.0, 0.0], [0.0, 1.0]])
    video_mu = np.array([[1.0, 0.0], [-1.0, 0.0]])
    text_logvar = np.log(np.full((2, 2), 0.25))
    video_logvar = np.log(np.full((2, 2), 0.5))

    summary = summarize_csd_components(
        text_mu, video_mu, text_logvar, video_logvar, chunk_size=1
    )

    assert summary["mean_distance"]["mean"] == 2.0
    assert summary["variance_distance"]["mean"] == 1.5
