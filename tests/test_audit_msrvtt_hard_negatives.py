from scripts.audit_msrvtt_hard_negatives import (
    AuditConfig,
    audit_and_clean_mapping,
    normalize_caption,
    pair_text_metrics,
)
from scripts.build_msrvtt_hard_negatives import CaptionSample


def sample(idx: int, video_id: str, caption: str) -> CaptionSample:
    return CaptionSample(
        sample_index=idx,
        video_id=video_id,
        caption=caption,
        sen_id=idx,
        json_sentence_index=idx,
    )


def mapping_item(anchor: int, hard: int, dense_score: float = 0.4) -> dict:
    return {
        "anchor_index": anchor,
        "anchor_video_id": f"video{anchor}",
        "anchor_sen_id": anchor,
        "anchor_json_sentence_index": anchor,
        "hard_index": hard,
        "hard_video_id": f"video{hard}",
        "hard_sen_id": hard,
        "hard_json_sentence_index": hard,
        "dense_rank": 10,
        "bm25_rank": 50,
        "dense_score": dense_score,
        "bm25_score": 8.0,
        "candidate_count": 500,
        "raw_posting_hits": 1000,
        "skipped_query_terms": 0,
    }


def test_pair_text_metrics_detects_exact_and_near_duplicate_captions():
    config = AuditConfig()

    assert normalize_caption("A Person, is Playing a VIDEO game!") == "a person is playing a video game"

    exact = pair_text_metrics(
        "a person is playing a video game",
        "a person is playing a video game",
        config,
    )
    near = pair_text_metrics(
        "a person is playing a racing video game",
        "playing a racing video game",
        config,
    )

    assert exact["exact_caption"] is True
    assert exact["jaccard"] == 1.0
    assert exact["overlap"] == 1.0
    assert near["exact_caption"] is False
    assert near["overlap"] >= 0.8


def test_audit_and_clean_mapping_filters_false_negatives_and_caps_reused_hard_caption():
    samples = [
        sample(0, "video0", "a person is playing a video game"),
        sample(1, "video1", "a person is playing a video game"),
        sample(2, "video2", "a woman is cooking food"),
        sample(3, "video3", "a dog runs outside"),
        sample(4, "video4", "a man sings on stage"),
        sample(5, "video5", "a child opens a present"),
    ]
    mapping = {
        "0": mapping_item(0, 1, dense_score=1.0),
        "2": mapping_item(2, 3, dense_score=0.35),
        "4": mapping_item(4, 3, dense_score=0.30),
        "5": mapping_item(5, 3, dense_score=0.25),
    }
    config = AuditConfig(
        clean_max_dense_score=0.95,
        clean_max_overlap=0.9,
        max_per_hard_index=2,
        max_per_hard_video=10,
    )

    clean_mapping, summary, rows = audit_and_clean_mapping(samples, mapping, config)

    assert "0" not in clean_mapping
    assert set(clean_mapping) == {"4", "5"}
    assert summary["clean"]["kept_mapping_size"] == 2
    assert summary["clean"]["removed_total"] == 2
    assert summary["clean"]["removal_reasons"]["exact_caption"] == 1
    assert summary["clean"]["removal_reasons"]["cap_hard_index"] == 1
    assert any(row["bucket"] == "removed" and "exact_caption" in row["reasons"] for row in rows)
