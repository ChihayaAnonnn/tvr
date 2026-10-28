"""Fixed CLIP ranking and bound natural/controlled candidate panels for E0/V0E."""

import copy
from pathlib import Path

import numpy as np
import torch

from experiment_artifacts import artifact_json_bytes, strict_json_loads
from semantic_retrieval.data.episodes import EpisodePreprocessing, file_sha256, read_episodes
from semantic_retrieval.data.tokenizer import ClipTokenizer
from semantic_retrieval.data.video import decode_video, sample_frame_indices
from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.runtime.input_bindings import context_input_references
from semantic_retrieval.runtime.pilot_protocol import _marker, validate_pilot_gpu_qualification

KS = (4, 8, 16, 32, 64, 128)


def file_reference(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=file_sha256(path))


def write_json(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(artifact_json_bytes(payload))


def fixed_clip_orders(query_features, video_features, canonical_ids):
    """Cosine ranking; stable canonical order is the sole tie breaker."""
    q, v = np.asarray(query_features), np.asarray(video_features)
    if (
        q.ndim != 2
        or v.ndim != 2
        or q.shape[1] != v.shape[1]
        or len(v) != len(canonical_ids)
        or tuple(canonical_ids) != tuple(sorted(set(canonical_ids)))
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or (np.linalg.norm(q, axis=1) == 0).any()
        or (np.linalg.norm(v, axis=1) == 0).any()
    ):
        raise ValueError("CLIP features require finite nonzero vectors and unique sorted canonical IDs")
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.argsort(-(q @ v.T), axis=1, kind="stable").astype(np.int32)


def _validate_orders(query_ids, canonical_ids, ranks, positives):
    ranks = np.asarray(ranks)
    if (
        len(set(query_ids)) != len(query_ids)
        or ranks.shape != (len(query_ids), len(canonical_ids))
        or len(canonical_ids) < 129
        or tuple(canonical_ids) != tuple(sorted(set(canonical_ids)))
        or any(not np.array_equal(np.sort(row), np.arange(len(canonical_ids))) for row in ranks)
        or set(query_ids) != set(positives)
        or any(len(positives[q]) != 1 or positives[q][0] not in canonical_ids for q in query_ids)
    ):
        raise ValueError(
            "fixed panels require complete canonical ranks, one official positive and at least 128 negatives"
        )


def panel_groups(query_ids, canonical_ids, ranks, positives):
    _validate_orders(query_ids, canonical_ids, ranks, positives)
    column = {i: j for j, i in enumerate(canonical_ids)}
    groups = []
    for track in ("natural", "controlled"):
        for k in KS:
            names, masks, initial = [], [], []
            for q, order in zip(query_ids, ranks):
                gt = column[positives[q][0]]
                neg = [int(i) for i in order if i != gt]
                panels = [order[:k]] if track == "natural" else [neg[: k - 1] + [gt], neg[:k]]
                first = [order[:4]] if track == "natural" else [neg[:3] + [gt], neg[:4]]
                for selected, start in zip(panels, first):
                    mask, init = np.zeros(len(canonical_ids), dtype=bool), np.zeros(len(canonical_ids), dtype=bool)
                    mask[selected], init[start] = True, True
                    names.append(q)
                    masks.append(mask)
                    initial.append(init)
            groups.append(
                dict(
                    track=track,
                    K=k,
                    arrays=dict(
                        query_ids=np.asarray(names),
                        candidate_mask=np.stack(masks),
                        initial_candidate_mask=np.stack(initial),
                    ),
                )
            )
    return groups


def ranked_episode_payload(payload, gallery_payload, ranks, positives):
    """Replace C1's software inventory with the true CLIP top128 negatives + GT."""
    result = copy.deepcopy(payload)
    members = {}
    for row in gallery_payload["videos"]:
        members.setdefault(row["canonical_id"], row)
    ids = tuple(sorted(members))
    qids = tuple(row["query_id"] for row in result["episodes"])
    _validate_orders(qids, ids, ranks, positives)
    sources = {}
    for original in payload["episodes"]:
        for canonical, source in zip(original["canonical_ids"], original.get("source_ids", [])):
            if canonical in sources and sources[canonical] != source:
                raise ValueError("original canonical source lineage conflicts across queries")
            sources[canonical] = source
    orders = {}
    for row, rank in zip(result["episodes"], ranks):
        gt = positives[row["query_id"]][0]
        negatives = [ids[i] for i in rank if ids[i] != gt][:128]
        selected = set(negatives + [gt])
        orders[row["query_id"]] = [ids[i] for i in rank if ids[i] in selected]
        inventory = sorted(selected)
        row.update(
            canonical_ids=inventory,
            candidate_ids=[members[i]["alias_id"] for i in inventory],
            relevance=["positive" if i == gt else "negative" for i in inventory],
            answer_in_candidates="yes",
            answer_in_gallery="yes",
            video_paths=[members[i]["video_path"] for i in inventory],
            content_sha256=[members[i]["content_sha256"] for i in inventory],
            frame_counts=[members[i]["frame_count"] for i in inventory],
            source_ids=[sources.get(i, members[i]["alias_id"]) for i in inventory],
        )
    return result, orders


def validate_candidate_config(config, *, phase_id):
    wanted = {"train", "val_development", "val_calibration"} if phase_id == "E0" else {"val_locked"}
    if phase_id not in {"E0", "V0E"}:
        raise ValueError("candidate producer requires E0 or V0E phase")
    if (
        not isinstance(config, dict)
        or set(config) != {"schema_version", "clip", "weights", "splits"}
        or config["schema_version"] != 1
        or set(config["splits"]) != wanted
        or any(set(v) != {"episodes", "gallery"} for v in config["splits"].values())
    ):
        raise ValueError("candidate producer splits differ from the phase; locked is sealed before V0E")


def candidate_manifest_config(config, *, phase_id):
    import hashlib

    validate_candidate_config(config, phase_id=phase_id)
    return dict(
        model={},
        loss={},
        optimizer={},
        scheduler={},
        evaluation=dict(
            eval_split="val_development" if phase_id == "E0" else "val_locked",
            qualification_scenario="pilot_candidates:" + hashlib.sha256(artifact_json_bytes(config)).hexdigest(),
        ),
    )


def _bound(context, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("candidate input requires an immutable file reference")
    path = Path(reference["path"])
    refs = context_input_references(context)
    if not path.is_absolute() or path.resolve(strict=True) != path or refs.get(str(path)) != reference["sha256"]:
        raise ValueError("candidate input must be a canonical reserved input")
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("candidate input hash differs from reservation")
    return path


def _gallery_rows(context, reference, preprocessing):
    path = _bound(context, reference)
    payload = strict_json_loads(path.read_text())
    if (
        set(payload) != {"schema_version", "gallery_version", "preprocessing", "videos"}
        or payload["schema_version"] != 1
        or payload["preprocessing"] != preprocessing
    ):
        raise ValueError("CLIP gallery schema/preprocessing mismatch")
    canonical, aliases = {}, set()
    for row in payload["videos"]:
        if (
            set(row) != {"canonical_id", "alias_id", "video_path", "content_sha256", "frame_count"}
            or row["alias_id"] in aliases
            or type(row["frame_count"]) is not int
            or row["frame_count"] <= 0
        ):
            raise ValueError("CLIP gallery contains an invalid or duplicate alias")
        _bound(context, dict(path=row["video_path"], sha256=row["content_sha256"]))
        old = canonical.get(row["canonical_id"])
        if old and (old["content_sha256"], old["frame_count"]) != (row["content_sha256"], row["frame_count"]):
            raise ValueError("canonical CLIP gallery content conflict")
        canonical.setdefault(row["canonical_id"], row)
        aliases.add(row["alias_id"])
    return payload, tuple(sorted(canonical)), canonical


@torch.no_grad()
def produce_candidate_artifacts(context, config, *, device):
    """Actual E0/V0E worker; no learned coarse projector or held-out selection."""
    phase = _marker(context.manifest)["phase_id"]
    validate_candidate_config(config, phase_id=phase)
    if context.manifest["config"] != candidate_manifest_config(config, phase_id=phase):
        raise ValueError("candidate configuration identity differs from reserved manifest")
    if device.type == "cuda":
        validate_pilot_gpu_qualification(context)
    clip_cfg = ClipConfig(**config["clip"])
    if device.type == "cuda" and clip_cfg.fixture:
        raise ValueError("scientific candidate ranking requires production ViT-B/32")
    weights = _bound(context, config["weights"])
    # Validate every raw/label input before allocating the model or scoring queries.
    inputs = {}
    for split, refs in config["splits"].items():
        path = _bound(context, refs["episodes"])
        payload = strict_json_loads(path.read_text())
        if payload.get("schema_version") != 2 or payload.get("relevance_space") != "benchmark_annotation":
            raise ValueError("candidate producer requires new v2 benchmark labels")
        for ref in payload["label_references"].values():
            _bound(context, ref)
        episodes = read_episodes(path, gallery_version=payload["gallery_version"], split=split)
        gallery, ids, rows = _gallery_rows(context, refs["gallery"], payload["preprocessing"])
        if gallery["gallery_version"] != payload["gallery_version"]:
            raise ValueError("candidate episode/gallery version mismatch")
        source_ids = {c for row in payload["episodes"] for c in row["canonical_ids"]}
        if source_ids != set(ids):
            raise ValueError("candidate input inventories must preserve source lineage for every gallery canonical ID")
        pre = EpisodePreprocessing(**payload["preprocessing"])
        if (
            pre.image_size != clip_cfg.image_size
            or pre.max_frames != 8
            or (not clip_cfg.fixture and pre.max_text_length != 32)
        ):
            raise ValueError("candidate producer preprocessing differs from frozen eight-frame CLIP protocol")
        inputs[split] = (payload, episodes, gallery, ids, rows, pre)
    clip = ClipBackbone(clip_cfg).to(device).eval().requires_grad_(False)
    load_clip_state(weights, config["weights"]["sha256"], clip)
    tokenizer, output = ClipTokenizer(), {}
    for split, (payload, episodes, gallery, ids, rows, pre) in inputs.items():
        features = []
        for i in ids:
            row = rows[i]
            frames, valid = decode_video(
                Path(row["video_path"]),
                frame_indices=sample_frame_indices(row["frame_count"], pre.max_frames),
                image_size=pre.image_size,
            )
            video, mask = frames[None].to(device), valid[None].to(device)
            encoded = clip.encode_frames(video, mask)
            features.append((encoded * mask[..., None]).sum(1).div(mask.sum(1, keepdim=True)).cpu().numpy()[0])
        video_features = np.stack(features)
        orders = []
        for start in range(0, len(episodes), 32):
            ids_batch, mask = tokenizer.encode_batch(
                [e.query_text for e in episodes[start : start + 32]], pre.max_text_length
            )
            _, pooled = clip.encode_text(ids_batch.to(device), mask.to(device))
            orders.append(fixed_clip_orders(pooled.cpu().numpy(), video_features, ids))
        ranks = np.concatenate(orders)
        qids = tuple(e.query_id for e in episodes)
        positives = {e.query_id: e.benchmark_positive_ids for e in episodes}
        directory = context.paths.provenance / "candidates" / split
        directory.mkdir(parents=True)
        rank_path = context.paths.arrays / f"{split}-clip-ranks.npz"
        with rank_path.open("xb") as stream:
            np.savez(stream, query_ids=np.asarray(qids), canonical_ids=np.asarray(ids), ranks=ranks)
        updated, train_orders = ranked_episode_payload(payload, gallery, ranks, positives)
        # C1 exports canonical absolute references; retain them across run-local consumers.
        for ref in updated["label_references"].values():
            ref["path"] = str(Path(ref["path"]).resolve())
        episode_path, order_path = directory / "episodes.json", directory / "orders.json"
        write_json(episode_path, updated)
        write_json(order_path, train_orders)
        groups, panel_refs = [], []
        if split != "train":
            for group in panel_groups(qids, ids, ranks, positives):
                path = context.paths.arrays / f"{split}-{group['track']}-k{group['K']}.npz"
                with path.open("xb") as stream:
                    np.savez(stream, **group["arrays"])
                groups.append(dict(track=group["track"], K=group["K"], arrays=file_reference(path)))
            for seed in (0, 1, 2):
                for arm in ("A0", "A1", "A2", "A3", "A4"):
                    path = directory / f"{arm}-seed{seed}-panels.json"
                    write_json(
                        path,
                        dict(
                            schema_version=1,
                            arm=arm,
                            seed=seed,
                            gallery_manifest_sha256=config["splits"][split]["gallery"]["sha256"],
                            split_sha256=file_sha256(episode_path),
                            canonical_ids=list(ids),
                            groups=groups,
                        ),
                    )
                    panel_refs.append(dict(arm=arm, seed=seed, panels=file_reference(path)))
        output[split] = dict(
            episodes=file_reference(episode_path),
            orders=file_reference(order_path),
            gallery=config["splits"][split]["gallery"],
            ranks=file_reference(rank_path),
            panels=panel_refs,
            array_inputs=[g["arrays"] for g in groups],
            label_references=updated["label_references"],
        )
    result = dict(
        schema_version=1,
        mechanism="fixed_clip_cosine_canonical_ties",
        phase_id=phase,
        source_sha=context.manifest["git_sha"],
        clip=config["clip"],
        clip_weights=config["weights"],
        splits=output,
    )
    write_json(context.paths.provenance / "pilot-candidates.json", result)
    return result
