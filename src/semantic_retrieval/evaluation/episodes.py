"""Episode metrics with explicit event denominators and incomplete-label counts."""

from dataclasses import dataclass

from semantic_retrieval.types import Decision


@dataclass(frozen=True)
class EpisodePrediction:
    query_id: str
    decision: Decision
    candidate_ids: tuple[str, ...]
    initial_candidate_ids: tuple[str, ...]
    attempted_k: tuple[int, ...]
    budget_used: int


def evaluate_episodes(predictions, targets, *, split):
    if split not in {"val", "val_development", "val_calibration", "val_locked"}:
        raise ValueError("evaluation requires validation; test is sealed")
    if len(predictions) != len(targets):
        raise ValueError("prediction/target length mismatch")
    names = (
        "coverage",
        "false_accept",
        "false_abstain",
        "set_precision",
        "set_recall",
        "expansion_recovery",
        "abstain_rate",
        "clarify_rate",
    )
    counts = {k: [0, 0] for k in names}
    types = {k: 0 for k in ("single_positive", "multiple_positive", "gallery_absent", "unknown", "initial_miss")}
    incomplete = 0
    total_budget = 0
    seen = set()
    for pred, target in zip(predictions, targets):
        if pred.query_id != target.query_id or pred.query_id in seen:
            raise ValueError("query identity mismatch or duplicate")
        seen.add(pred.query_id)
        truth = dict(zip(target.canonical_ids, target.relevance))
        positive = {i for i, label in truth.items() if label == 1}
        if (
            len(truth) != len(target.canonical_ids)
            or len(target.canonical_ids) != len(target.relevance)
            or any(label not in {-1, 0, 1} for label in truth.values())
        ):
            raise ValueError("invalid evaluation labels")
        if (
            type(pred.budget_used) is not int
            or pred.budget_used < 0
            or not set(pred.decision.answer_ids) <= set(pred.candidate_ids)
            or len(set(pred.candidate_ids)) != len(pred.candidate_ids)
        ):
            raise ValueError("invalid prediction budget/candidates")
        returned = pred.decision.action == "return"
        abstained = pred.decision.action == "abstain"
        for key, event in [
            ("coverage", returned),
            ("abstain_rate", abstained),
            ("clarify_rate", pred.decision.action == "clarify"),
        ]:
            counts[key][0] += int(event)
            counts[key][1] += 1
        total_budget += pred.budget_used
        gallery = target.answer_in_gallery
        if gallery not in {-1, 0, 1}:
            raise ValueError("invalid gallery presence label")
        if gallery == 0:
            if positive:
                raise ValueError("gallery absence conflicts with positive labels")
            types["gallery_absent"] += 1
            counts["false_accept"][0] += int(returned)
            counts["false_accept"][1] += 1
        if gallery == 1 or positive:
            counts["false_abstain"][0] += int(abstained)
            counts["false_abstain"][1] += 1
        if positive:
            types["multiple_positive" if len(positive) > 1 else "single_positive"] += 1
        elif gallery == -1:
            types["unknown"] += 1
        complete = all(label >= 0 for label in truth.values()) and all(i in truth for i in pred.decision.answer_ids)
        if not complete:
            incomplete += 1
        else:
            answers = set(pred.decision.answer_ids) if returned else set()
            tp = len(answers & positive)
            counts["set_precision"][0] += tp
            counts["set_precision"][1] += len(answers)
            counts["set_recall"][0] += tp
            counts["set_recall"][1] += len(positive)
            miss = bool(positive) and not (set(pred.initial_candidate_ids) & positive)
            if miss:
                types["initial_miss"] += 1
                counts["expansion_recovery"][0] += int(bool(answers & positive))
                counts["expansion_recovery"][1] += 1
    metrics = {k: dict(numerator=n, denominator=d, value=n / d if d else None) for k, (n, d) in counts.items()}
    metrics.update(
        episode_count=len(predictions),
        episode_types=types,
        incomplete_label_episodes=incomplete,
        budget=dict(total=total_budget, mean=total_budget / len(predictions) if predictions else None),
    )
    return metrics
