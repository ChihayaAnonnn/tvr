"""Pack canonical episode candidates with explicit unknown cross-query relations."""

import torch

from semantic_retrieval.types import EncoderInputs, EpisodeBatch, Targets

from .episodes import audit_episode_splits, file_sha256
from .tokenizer import ClipTokenizer
from .video import decode_video, sample_frame_indices


def collate_episodes(episodes):
    episodes = list(episodes)
    if not episodes:
        raise ValueError("cannot collate an empty query batch")
    audit_episode_splits(episodes)
    if len({e.query_id for e in episodes}) != len(episodes):
        raise ValueError("query IDs must be unique")
    preprocessing = episodes[0].preprocessing
    if any(e.preprocessing != preprocessing or e.gallery_version != episodes[0].gallery_version for e in episodes):
        raise ValueError("batch preprocessing/gallery mismatch")
    candidates = {}
    for e in episodes:
        for i, canonical in enumerate(e.canonical_ids):
            item = (e.video_paths[i], e.content_sha256[i], e.frame_counts[i])
            if canonical in candidates and candidates[canonical][1:] != item[1:]:
                raise ValueError("canonical content/frame-count conflict")
            candidates.setdefault(canonical, item)
    ids = tuple(sorted(candidates))
    lookup = {canonical: i for i, canonical in enumerate(ids)}
    video = torch.zeros(len(ids), preprocessing.max_frames, 3, preprocessing.image_size, preprocessing.image_size)
    mask = torch.zeros(len(ids), preprocessing.max_frames, dtype=torch.bool)
    for canonical, i in lookup.items():
        path, digest, count = candidates[canonical]
        if file_sha256(path) != digest:
            raise ValueError("candidate content changed before collation")
        indices = sample_frame_indices(count, preprocessing.max_frames)
        frames, valid = decode_video(path, indices, preprocessing.image_size)
        video[i, : len(indices)] = frames
        mask[i, : len(indices)] = valid
    input_ids, text_mask = ClipTokenizer().encode_batch([e.query_text for e in episodes], preprocessing.max_text_length)
    labels = torch.full((len(episodes), len(ids)), -1, dtype=torch.long)
    pairs = []
    offsets = [0]
    group_pairs = []
    group_labels = []
    group_provenance = []
    for q, e in enumerate(episodes):
        row_pair = {}
        for canonical, label in zip(e.canonical_ids, e.relevance):
            labels[q, lookup[canonical]] = label
            row_pair[canonical] = len(pairs)
            pairs.append((q, lookup[canonical]))
        offsets.append(len(pairs))
        for a, b, label in e.query_conditioned_group_labels:
            group_pairs.append((row_pair[a], row_pair[b]))
            group_labels.append(label)
            group_provenance.append(e.annotation_provenance["sha256"])
    targets = Targets(
        labels,
        torch.tensor([e.answer_in_candidates for e in episodes]),
        torch.tensor(group_pairs, dtype=torch.long).reshape(-1, 2) if group_pairs else None,
        torch.tensor(group_labels, dtype=torch.long) if group_pairs else None,
    )
    return EpisodeBatch(
        EncoderInputs(input_ids, text_mask, video, mask),
        targets,
        tuple(e.query_id for e in episodes),
        ids,
        torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        torch.tensor(offsets, dtype=torch.long),
        tuple(group_provenance) if group_provenance else None,
    )
