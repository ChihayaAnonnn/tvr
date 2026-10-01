#!/usr/bin/env python3
"""Reserved, newly qualified GPU export of fixed-panel benchmark model scores."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from semantic_retrieval.config import load_system_config
from semantic_retrieval.data.episodes import read_episodes
from semantic_retrieval.data.gallery import audit_episode_gallery, preprocessing_sha256
from semantic_retrieval.data.tokenizer import ClipTokenizer
from semantic_retrieval.evaluation.pilot_analysis import (
    _read_hashed,
    export_model_scores,
    file_sha256,
    stream_pilot_gallery,
)
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.checkpoint import load_model_checkpoint
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.inference import policy_sha256
from semantic_retrieval.types import QueryInputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--panels-sha256", required=True)
    args = parser.parse_args(argv)
    configure_fp32()
    cfg = load_system_config(args.config)
    if (
        cfg.data is None
        or cfg.deployment is None
        or cfg.data.split not in ("val_development", "val_calibration", "val_locked")
        or cfg.runtime.eval_split != cfg.data.split
    ):
        raise ValueError("explicit benchmark validation score export configuration required")
    if getattr(cfg.data, "relevance_space", None) != "benchmark_annotation" or cfg.model.pilot_arm not in (
        "A0",
        "P1",
        "A1",
        "A2",
        "A3",
        "A4",
    ):
        raise ValueError("benchmark v2 and explicit pilot arm required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available():
        raise ValueError("scientific score export requires one process on a qualified CUDA device")
    context = prepare_run(cfg.runtime, repo_root=Path(__file__).resolve().parents[1], system_config=cfg)
    # The new study must prove its own Q0; old four-card reports cannot authorize this worker.
    from semantic_retrieval.runtime.pilot_protocol import validate_pilot_gpu_qualification

    validate_pilot_gpu_qualification(context)
    refs = {r["path"]: r["sha256"] for r in context.manifest["dataset_splits"]}
    panels_path = _read_hashed(dict(path=str(args.panels.resolve()), sha256=args.panels_sha256))
    if refs.get(str(panels_path)) != args.panels_sha256:
        raise ValueError("E0 panel manifest must be a reserved input")
    panel_manifest = json.loads(panels_path.read_text())
    if (
        panel_manifest.get("schema_version") != 1
        or panel_manifest.get("arm") != cfg.model.pilot_arm
        or panel_manifest.get("seed") != cfg.train.seed
    ):
        raise ValueError("panel manifest arm/seed identity mismatch")
    if sorted((g["track"], g["K"]) for g in panel_manifest["groups"]) != sorted(
        (track, k) for track in ("natural", "controlled") for k in (4, 8, 16, 32, 64, 128)
    ):
        raise ValueError("all six natural and balanced controlled panels required")
    panels = []
    for group in panel_manifest["groups"]:
        if refs.get(group["arrays"]["path"]) != group["arrays"]["sha256"]:
            raise ValueError("E0 panel NPZ must be reserved")
        with np.load(_read_hashed(group["arrays"]), allow_pickle=False) as arrays:
            panels.append(dict(track=group["track"], K=group["K"], **{k: arrays[k] for k in arrays.files}))
    start_run(context)
    try:
        episodes = read_episodes(
            cfg.data.episode_manifest, gallery_version=cfg.data.gallery_version, split=cfg.data.split
        )
        if any(
            e.relevance_space != "benchmark_annotation" or dict(e.label_protocol) != dict(cfg.data.label_protocol)
            for e in episodes
        ):
            raise ValueError("episode benchmark scope differs from score export contract")
        source = json.loads(cfg.data.episode_manifest.read_text())
        relation_ref = source["label_references"]["relations"]
        relation_path = Path(relation_ref["path"])
        if not relation_path.is_absolute():
            relation_path = cfg.data.episode_manifest.parent / relation_path
        if refs.get(str(relation_path.resolve())) != relation_ref["sha256"]:
            raise ValueError("official relation table must be a reserved input")
        relations = json.loads(
            _read_hashed(dict(path=str(relation_path.resolve()), sha256=relation_ref["sha256"])).read_text()
        )
        gallery = stream_pilot_gallery(cfg.deployment.gallery_manifest, gallery_version=cfg.data.gallery_version)
        audit_episode_gallery(episodes, gallery)
        if (
            panel_manifest.get("gallery_manifest_sha256") != cfg.deployment.gallery_sha256
            or panel_manifest.get("split_sha256") != cfg.data.episode_sha256
            or panel_manifest.get("canonical_ids") != list(gallery.canonical_ids)
        ):
            raise ValueError("E0 panel gallery/split/column identity mismatch")
        gallery_payload = dict(
            version=cfg.data.gallery_version,
            manifest_sha256=cfg.deployment.gallery_sha256,
            canonical_ids=list(gallery.canonical_ids),
            aliases={c: list(gallery.aliases[c]) for c in gallery.canonical_ids},
        )
        query_ids = [e.query_id for e in episodes]
        official_ids = [v for c in gallery.canonical_ids for v in gallery.aliases[c]]
        canonical = np.array(
            [[c in e.benchmark_positive_ids for c in gallery.canonical_ids] for e in episodes], np.int8
        )
        official = np.array(
            [[v in relations["official_positives_by_query"][e.query_id] for v in official_ids] for e in episodes],
            np.int8,
        )
        for panel in panels:
            indices = {q: i for i, q in enumerate(query_ids)}
            y = np.array(
                [
                    (canonical[indices[str(q)]].astype(bool) & mask).any()
                    for q, mask in zip(panel["query_ids"], panel["candidate_mask"])
                ]
            )
            if panel["track"] == "controlled" and (y.sum() * 2 != len(y)):
                raise ValueError("controlled calibration/evaluation panels must remain balanced")
        tokenizer = ClipTokenizer()
        ids, mask = tokenizer.encode_batch([e.query_text for e in episodes], gallery.preprocessing.max_text_length)
        model = SemanticRetrievalModel(cfg.model).to("cuda:0")
        load_model_checkpoint(
            cfg.deployment.checkpoint_path, model, expected_config_sha256=cfg.deployment.checkpoint_config_sha256
        )
        fragment = export_model_scores(
            context,
            model,
            QueryInputs(ids.to("cuda:0"), mask.to("cuda:0")),
            gallery.batches(),
            None,
            query_ids=query_ids,
            component_ids=[e.query_component_id for e in episodes],
            canonical_relevance=canonical,
            official_relevance=official,
            gallery=gallery_payload,
            panels=panels,
            arm=cfg.model.pilot_arm,
            seed=cfg.train.seed,
            split=cfg.data.split,
            split_sha256=cfg.data.episode_sha256,
            label_protocol=dict(cfg.data.label_protocol),
            preprocessing_sha256=preprocessing_sha256(gallery.preprocessing),
            policy_sha256=policy_sha256(cfg.policy),
        )
        path = context.paths.provenance / "pilot_score_fragment.json"
        publish_metrics(
            context,
            dict(
                score_fragment_reference=dict(
                    path=path.relative_to(context.paths.root).as_posix(), sha256=file_sha256(path)
                ),
                group_count=len(fragment["groups"]),
                evaluation_gpu_hours=fragment["evaluation_gpu_hours"],
                array_references=[],
            ),
        )
        complete_run(context)
    except BaseException:
        fail_run(context, exit_code=1)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
