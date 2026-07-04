from scripts.build_msrvtt_model_mined_hard_negatives import (
    CaptionSample,
    ModelHardNegativeConfig,
    build_model_mined_mapping,
    select_hard_negative_candidate,
    select_query_samples,
    text_pair_metrics,
)


def test_select_hard_negative_candidate_skips_same_video_and_near_duplicates():
    samples = [
        CaptionSample(0, "video0", "a man plays guitar"),
        CaptionSample(1, "video1", "a man plays guitar"),
        CaptionSample(2, "video0", "same source video"),
        CaptionSample(3, "video3", "a woman cooks pasta"),
        CaptionSample(4, "video4", "children play soccer"),
    ]
    video_to_sample_indices = {
        "video0": [0, 2],
        "video1": [1],
        "video3": [3],
        "video4": [4],
    }
    config = ModelHardNegativeConfig(top_k=4, min_rank=1, max_jaccard=0.8, max_overlap=0.9)

    candidate = select_hard_negative_candidate(
        anchor=samples[0],
        ranked_video_indices=[0, 1, 2, 3],
        video_ids=["video0", "video1", "video3", "video4"],
        samples=samples,
        video_to_sample_indices=video_to_sample_indices,
        config=config,
    )

    assert candidate is not None
    assert candidate.hard_index == 3
    assert candidate.hard_video_id == "video3"
    assert candidate.model_rank == 3
    assert candidate.skipped_same_video == 1
    assert candidate.skipped_text_risk == 1


def test_select_hard_negative_candidate_respects_min_rank_window():
    samples = [
        CaptionSample(0, "video0", "a man plays guitar"),
        CaptionSample(1, "video1", "a woman cooks pasta"),
        CaptionSample(2, "video2", "children play soccer"),
    ]
    video_to_sample_indices = {"video0": [0], "video1": [1], "video2": [2]}
    config = ModelHardNegativeConfig(top_k=2, min_rank=2)

    candidate = select_hard_negative_candidate(
        anchor=samples[0],
        ranked_video_indices=[1, 2],
        video_ids=["video0", "video1", "video2"],
        samples=samples,
        video_to_sample_indices=video_to_sample_indices,
        config=config,
    )

    assert candidate is not None
    assert candidate.hard_index == 2
    assert candidate.model_rank == 2


def test_text_pair_metrics_flags_exact_and_overlap():
    metrics = text_pair_metrics("A man plays guitar on stage", "man playing guitar on a stage")

    assert metrics["shared_tokens"] >= 2
    assert metrics["jaccard"] > 0.25
    assert metrics["overlap"] >= 0.5
    assert metrics["exact_caption"] is False


def test_build_model_mined_mapping_writes_compatible_entries_and_stats():
    samples = [
        CaptionSample(0, "video0", "a man plays guitar"),
        CaptionSample(1, "video1", "a woman cooks pasta"),
        CaptionSample(2, "video2", "children play soccer"),
    ]
    video_ids = ["video0", "video1", "video2"]
    scores = [
        [10.0, 9.5, 3.0],
        [2.0, 8.0, 7.5],
        [5.0, 4.5, 9.0],
    ]
    config = ModelHardNegativeConfig(top_k=3, min_rank=1)

    result = build_model_mined_mapping(
        samples=samples,
        video_ids=video_ids,
        scores=scores,
        config=config,
        include_captions=True,
    )

    assert result["stats"]["num_samples"] == 3
    assert result["stats"]["mapping_size"] == 3
    assert result["stats"]["fallback_count"] == 0
    assert result["mapping"]["0"]["anchor_index"] == 0
    assert result["mapping"]["0"]["hard_index"] == 1
    assert result["mapping"]["0"]["hard_video_id"] == "video1"
    assert result["mapping"]["0"]["model_rank"] == 2
    assert result["mapping"]["0"]["anchor_caption"] == "a man plays guitar"


def test_select_query_samples_uses_exclusive_range_then_limit():
    samples = [CaptionSample(i, f"video{i}", f"caption {i}") for i in range(10)]

    selected, query_start, query_end = select_query_samples(
        samples,
        query_start=3,
        query_end=8,
        limit_queries=2,
    )

    assert [sample.sample_index for sample in selected] == [3, 4]
    assert query_start == 3
    assert query_end == 5


def test_select_query_samples_rejects_invalid_ranges():
    samples = [CaptionSample(i, f"video{i}", f"caption {i}") for i in range(3)]

    try:
        select_query_samples(samples, query_start=2, query_end=1, limit_queries=0)
    except ValueError as exc:
        assert "query_end" in str(exc)
    else:
        raise AssertionError("expected invalid query range to fail")
