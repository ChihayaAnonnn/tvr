"""Hash-bound, nonrecursive file indices keep full galleries below the manifest size cap."""

import re
from pathlib import Path

from experiment_artifacts import artifact_json_bytes, strict_json_load
from semantic_retrieval.data.episodes import file_sha256

INDEX_SUFFIX = ".input-index.json"


def manifest_input_references(repository: Path, manifest: dict) -> dict[str, str]:
    """Expand authenticated indices; callers still verify every consumed file's bytes."""
    references = {}
    indices = []
    for field in ("dataset_splits", "input_checkpoints"):
        for ref in manifest[field]:
            name, digest = str(ref["path"]), ref["sha256"]
            if name in references and references[name] != digest:
                raise ValueError("conflicting reserved input identity")
            references[name] = digest
            if name.endswith(INDEX_SUFFIX):
                indices.append((name, digest))
    for name, digest in indices:
        path = Path(name)
        if not path.is_absolute():
            path = Path(repository) / path
        if path.is_symlink() or path.resolve() != path or file_sha256(path) != digest:
            raise ValueError("input index SHA/path mismatch")
        if path.stat().st_size >= 1024 * 1024:
            raise ValueError("input index must remain below 1 MiB")
        with path.open() as stream:
            payload = strict_json_load(stream)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "kind", "files"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["kind"] != "file_sha256_index"
            or not isinstance(payload["files"], list)
            or not payload["files"]
        ):
            raise ValueError("invalid input index contract")
        for item in payload["files"]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError("invalid indexed reference")
            child, sha = item["path"], item["sha256"]
            if (
                not isinstance(child, str)
                or not isinstance(sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                or child.endswith(INDEX_SUFFIX)
                or child in references
            ):
                raise ValueError("duplicate, nested or invalid indexed reference")
            target = Path(child)
            if not target.is_absolute() or target.is_symlink() or target.resolve() != target:
                raise ValueError("indexed input must have a canonical absolute path")
            references[child] = sha
    return references


def context_input_references(context) -> dict[str, str]:
    """A worker reuses its authenticated identity map instead of reparsing per pair."""
    if getattr(context, "input_references", None) is None:
        context.input_references = manifest_input_references(context.repo_root, context.manifest)
    return context.input_references


def write_input_indices(directory: Path, prefix: str, references: list[dict], *, max_files: int = 900) -> list[dict]:
    """Publish exclusive, small index shards; the caller chooses a permitted output directory."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", prefix) or not references or not 1 <= max_files <= 900:
        raise ValueError("invalid index writer parameters")
    results = []
    for offset in range(0, len(references), max_files):
        path = Path(directory) / f"{prefix}-{offset // max_files:03d}{INDEX_SUFFIX}"
        data = artifact_json_bytes(
            {"schema_version": 1, "kind": "file_sha256_index", "files": references[offset : offset + max_files]}
        )
        with path.open("xb") as stream:
            stream.write(data)
        results.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return results
