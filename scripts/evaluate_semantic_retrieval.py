#!/usr/bin/env python3
"""Reserved validation calibration/evaluation for the new retrieval system."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.config import load_system_config
from semantic_retrieval.data.episodes import file_sha256, read_episodes
from semantic_retrieval.data.gallery import audit_episode_gallery, preprocessing_sha256, read_gallery
from semantic_retrieval.data.tokenizer import ClipTokenizer
from semantic_retrieval.evaluation.calibration import CalibrationIdentity, CalibrationState, fit_calibration
from semantic_retrieval.evaluation.episodes import evaluate_episodes
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.retrieval.cache import FeatureCache
from semantic_retrieval.retrieval.index import CanonicalIndex, IndexIdentity
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.checkpoint import load_model_checkpoint, model_sha256
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.inference import RetrievalService, policy_sha256
from semantic_retrieval.types import EncodedFeatures, QueryInputs, VideoFeatures


def main(argv=None):
    configure_fp32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    for arg in ("run-id", "manifest-sha256", "source-sha"):
        parser.add_argument("--" + arg)
    args = parser.parse_args(argv)
    cfg = load_system_config(args.config)
    dep = cfg.deployment
    if dep is None or cfg.data is None or cfg.data.split == "train" or cfg.runtime.eval_split != cfg.data.split:
        raise ValueError("explicit validation deployment configuration required")
    if dep.mode == "calibrate" and cfg.data.split != "val_calibration":
        raise ValueError("fit only on val_calibration")
    for name in ("run_id", "manifest_sha256", "source_sha"):
        if getattr(args, name) is not None and getattr(args, name) != getattr(cfg.runtime, name):
            raise ValueError("CLI identity mismatch")
    context = prepare_run(cfg.runtime, repo_root=Path(__file__).resolve().parents[1], system_config=cfg)
    start_run(context)
    try:
        for path, digest in (
            (cfg.data.episode_manifest, cfg.data.episode_sha256),
            (dep.gallery_manifest, dep.gallery_sha256),
            (dep.checkpoint_path, dep.checkpoint_sha256),
        ):
            if file_sha256(path) != digest:
                raise ValueError("deployment input changed")
        episodes = read_episodes(
            cfg.data.episode_manifest, gallery_version=cfg.data.gallery_version, split=cfg.data.split
        )
        gallery = read_gallery(dep.gallery_manifest, gallery_version=cfg.data.gallery_version)
        audit_episode_gallery(episodes, gallery)
        model = SemanticRetrievalModel(cfg.model)
        load_model_checkpoint(dep.checkpoint_path, model, expected_config_sha256=dep.checkpoint_config_sha256)
        model.eval().requires_grad_(False)
        pre_sha = preprocessing_sha256(gallery.preprocessing)
        ident = IndexIdentity(model_sha256(model), cfg.data.gallery_version, pre_sha)
        with torch.no_grad():
            v = model.encode_gallery(gallery.video, gallery.frame_mask)
            emb = model.project_gallery(v)
        cache = FeatureCache(context)
        for j, i in enumerate(gallery.canonical_ids):
            cache.write(
                i,
                VideoFeatures(v.frame_tokens[j : j + 1], v.pooled_video[j : j + 1], v.frame_mask[j : j + 1]),
                identity=ident,
                content_sha256=gallery.content_sha256[i],
            )
        index = CanonicalIndex(gallery.canonical_ids, emb, ident)
        # All arrays remain run-local; JSON stores only hashes/references.
        index_ref = context.array_store.write("coarse_index", emb.cpu().numpy()).as_dict()
        cal_id = CalibrationIdentity(
            ident.model_sha256, pre_sha, ident.gallery_version, policy_sha256(cfg.policy), dep.calibration_split_sha256
        )
        tokenizer = ClipTokenizer()
        if dep.mode == "calibrate":
            if cfg.data.episode_sha256 != dep.calibration_split_sha256:
                raise ValueError("calibration split identity mismatch")
            pair_logits = []
            pair_labels = []
            presence_logits = []
            presence_labels = []
            for e in episodes:
                ids, mask = tokenizer.encode_batch([e.query_text], e.preprocessing.max_text_length)
                with torch.no_grad():
                    q = model.encode_query(QueryInputs(ids, mask))
                    qemb = model.project_query(q)
                    # Calibration covers each predeclared candidate budget, not GT-generated C.
                    for k in cfg.policy.k_sequence:
                        found = index.search(qemb, k, expected_identity=ident)[0]
                        positions = [gallery.canonical_ids.index(i) for i in found]
                        f = EncodedFeatures(
                            q.text_tokens,
                            q.pooled_text,
                            v.frame_tokens[positions],
                            v.pooled_video[positions],
                            q.text_mask,
                            v.frame_mask[positions],
                        )
                        pairs = torch.tensor([[0, j] for j in range(len(found))], dtype=torch.long).reshape(-1, 2)
                        scores = model.score_pairs(f, pairs)
                        presence = model.score_presence(scores, torch.tensor([0, len(found)]))
                        truth = dict(zip(e.canonical_ids, e.relevance))
                        labels = [truth.get(i, -1) for i in found]
                        plabel = 1 if 1 in labels else (0 if all(label >= 0 for label in labels) else -1)
                        pair_logits.extend(scores.logits.tolist())
                        pair_labels.extend(labels)
                        presence_logits.extend(presence.tolist())
                        presence_labels.append(plabel)
            cal = fit_calibration(
                torch.tensor(pair_logits),
                torch.tensor(pair_labels, dtype=torch.long),
                torch.tensor(presence_logits),
                torch.tensor(presence_labels, dtype=torch.long),
                split=cfg.data.split,
                identity=cal_id,
            )
            path = context.paths.provenance / "calibration.json"
            with path.open("xb") as handle:
                handle.write(artifact_json_bytes(asdict(cal)))
            for name, values in [
                ("pair_logits", pair_logits),
                ("pair_labels", pair_labels),
                ("presence_logits", presence_logits),
                ("presence_labels", presence_labels),
            ]:
                context.array_store.write(name, values)
            publish_metrics(context, dict(calibration=asdict(cal), array_references=[index_ref]))
        else:
            if file_sha256(dep.calibration_path) != dep.calibration_sha256:
                raise ValueError("calibration content changed")
            raw = json.loads(dep.calibration_path.read_text())
            raw["identity"] = CalibrationIdentity(**raw["identity"])
            cal = CalibrationState(**raw)
            if cal.identity != cal_id:
                raise ValueError("frozen calibration identity mismatch")
            service = RetrievalService(model, index, cache, cal, cfg.policy, content_sha256=gallery.content_sha256)
            predictions = []
            for e in episodes:
                ids, mask = tokenizer.encode_batch([e.query_text], e.preprocessing.max_text_length)
                service.retrieve(QueryInputs(ids, mask), budget=cfg.policy.max_budget)
                predictions.append(replace(service.last_prediction, query_id=e.query_id))
            metrics = evaluate_episodes(predictions, episodes, split=cfg.data.split)
            metrics["index"] = index_ref
            metrics["budget_reference"] = context.array_store.write(
                "candidate_budget", [p.budget_used for p in predictions]
            ).as_dict()
            metrics["array_references"] = [index_ref, metrics["budget_reference"]]
            publish_metrics(context, metrics)
        complete_run(context)
    except BaseException:
        fail_run(context, exit_code=1)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
