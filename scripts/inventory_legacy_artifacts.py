#!/usr/bin/env python3
"""Create a resumable, append-only inventory of read-only legacy artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
CLASSIFICATIONS = frozenset({"critical", "historical", "failed", "unresolved"})
_CHUNK_SIZE = 8 * 1024 * 1024


def _absolute(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = Path.cwd() / value
    return Path(os.path.normpath(value))


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlink is forbidden in legacy artifact path: {current}")


def _regular_metadata(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink is forbidden in legacy artifacts: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"legacy inventory accepts only regular files: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"hardlink is forbidden in legacy artifacts: {path}")
    return metadata


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _hash_fd(fd: int, chunk_size: int = _CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sample_fd(fd: int, size: int) -> str:
    """Cheaply detect same-size writes on filesystems with coarse timestamps."""
    digest = hashlib.sha256()
    width = min(64 * 1024, size)
    offsets = sorted({0, max(0, (size - width) // 2), max(0, size - width)})
    for offset in offsets:
        digest.update(os.pread(fd, width, offset))
    return digest.hexdigest()


def _stable_hash(path: Path, chunk_size: int = _CHUNK_SIZE) -> tuple[os.stat_result, str]:
    before_path = _regular_metadata(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before_fd = os.fstat(fd)
        if not _same_file(before_path, before_fd):
            raise ValueError(f"legacy source changed before hashing: {path}")
        before_sample = _sample_fd(fd, before_fd.st_size)
        digest = _hash_fd(fd, chunk_size)
        after_fd = os.fstat(fd)
        after_sample = _sample_fd(fd, after_fd.st_size)
    finally:
        os.close(fd)
    after_path = _regular_metadata(path)
    if (
        not _same_file(before_fd, after_fd)
        or not _same_file(before_fd, after_path)
        or before_sample != after_sample
    ):
        raise ValueError(f"legacy source changed while hashing: {path}")
    return after_path, digest


def _walk_regular_files(root: Path) -> Iterator[Path]:
    def visit(directory: Path) -> Iterator[Path]:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"symlink is forbidden in legacy artifacts: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                yield from visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise ValueError(f"hardlink is forbidden in legacy artifacts: {path}")
                yield path
            else:
                raise ValueError(f"special file is forbidden in legacy artifacts: {path}")

    yield from visit(root)


def _classification_for(path: Path, classifications: Mapping[str, str]) -> str:
    best: tuple[int, str] | None = None
    for raw_prefix, classification in classifications.items():
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"invalid legacy classification: {classification}")
        prefix = _absolute(raw_prefix)
        try:
            path.relative_to(prefix)
        except ValueError:
            continue
        candidate = (len(prefix.parts), classification)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best is not None else "historical"


def _record(root: Path, path: Path, classifications: Mapping[str, str]) -> dict[str, object]:
    metadata, digest = _stable_hash(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "relative_path": path.relative_to(root).as_posix(),
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "sha256": digest,
        "classification": _classification_for(path, classifications),
        "state": "completed",
    }


def _validate_record(value: object) -> dict[str, object]:
    fields = {
        "schema_version", "root", "relative_path", "size", "mtime_ns",
        "sha256", "classification", "state",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid legacy inventory record")
    if value["schema_version"] != SCHEMA_VERSION or value["state"] != "completed":
        raise ValueError("invalid legacy inventory completion state")
    if value["classification"] not in CLASSIFICATIONS:
        raise ValueError("invalid legacy inventory classification")
    relative = Path(str(value["relative_path"]))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("invalid legacy inventory relative path")
    if not isinstance(value["size"], int) or value["size"] < 0:
        raise ValueError("invalid legacy inventory size")
    if not isinstance(value["mtime_ns"], int) or value["mtime_ns"] < 0:
        raise ValueError("invalid legacy inventory mtime")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid legacy inventory SHA-256")
    return value


def _load_resume_records(output: Path) -> list[dict[str, object]]:
    if not output.exists():
        return []
    _assert_no_symlink_components(output)
    metadata = output.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("inventory output must be a private regular file")
    payload = output.read_bytes()
    complete_end = payload.rfind(b"\n") + 1
    complete, tail = payload[:complete_end], payload[complete_end:]
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        try:
            records.append(_validate_record(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid inventory JSONL line {line_number}") from exc
    if tail:
        try:
            _validate_record(json.loads(tail))
        except (json.JSONDecodeError, ValueError):
            with output.open("r+b") as handle:
                handle.truncate(complete_end)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            raise ValueError("inventory JSONL must end each completed record with newline")
    return records


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path.parent)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) >= 1024 * 1024:
        raise ValueError("inventory summary JSON must remain below 1 MiB")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def inventory_legacy_artifacts(
    roots: Iterable[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    summary: str | os.PathLike[str],
    *,
    classifications: Mapping[str, str] | None = None,
) -> dict[str, object]:
    canonical_roots = [_absolute(root) for root in roots]
    if not canonical_roots:
        raise ValueError("at least one legacy root is required")
    for root in canonical_roots:
        _assert_no_symlink_components(root)
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"legacy root must be a directory: {root}")
    output_path = _absolute(output)
    summary_path = _absolute(summary)
    prior = _load_resume_records(output_path)
    completed: list[dict[str, object]] = []
    prior_index = 0
    classification_map = classifications or {}

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink_components(output_path.parent)
    with output_path.open("ab", buffering=0) as handle:
        for root in canonical_roots:
            for path in _walk_regular_files(root):
                relative = path.relative_to(root).as_posix()
                if prior_index < len(prior):
                    existing = prior[prior_index]
                    if existing["root"] != str(root) or existing["relative_path"] != relative:
                        raise ValueError("inventory resume records are not a prefix of the legacy roots")
                    if prior_index == len(prior) - 1:
                        metadata, digest = _stable_hash(path)
                        if (
                            existing["size"] != metadata.st_size
                            or existing["mtime_ns"] != metadata.st_mtime_ns
                            or existing["sha256"] != digest
                        ):
                            raise ValueError("last verified inventory record no longer matches its source")
                    completed.append(existing)
                    prior_index += 1
                    continue
                value = _record(root, path, classification_map)
                encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(encoded)
                os.fsync(handle.fileno())
                completed.append(value)
    if prior_index != len(prior):
        raise ValueError("inventory resume contains records absent from the legacy roots")

    inventory_bytes = output_path.read_bytes()
    counts = Counter(str(record["classification"]) for record in completed)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "roots": [str(root) for root in canonical_roots],
        "inventory_path": str(output_path),
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "completed_files": len(completed),
        "completed_bytes": sum(int(record["size"]) for record in completed),
        "classification_counts": dict(sorted(counts.items())),
        "state": "completed",
    }
    _atomic_json(summary_path, result)
    return result


def _mapping_classifications(mapping_path: Path, source_root: Path) -> dict[str, str]:
    if not mapping_path.exists():
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for run in payload.get("runs", []):
        classification = run.get("classification", "unresolved")
        for source in run.get("sources", []):
            relative = Path(source["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("mapping source path must be relative and traversal-free")
            result[str(source_root / relative)] = classification
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--mapping")
    parser.add_argument("--source-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = _absolute(args.source_root)
    classifications = (
        _mapping_classifications(_absolute(args.mapping), source_root)
        if args.mapping else {}
    )
    result = inventory_legacy_artifacts(
        args.roots, args.output, args.summary, classifications=classifications
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
