#!/usr/bin/env python3
"""Copy a strict whitelist of legacy artifacts into verified run directories."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterable, Mapping, Sequence


_CHUNK_SIZE = 8 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_STAGES = frozenset(
    {
        "baseline", "feature_ablation", "loss_ablation", "topk_sweep",
        "multiseed", "calibration", "robustness", "efficiency", "final",
    }
)


@dataclass(frozen=True)
class MigrationResult:
    source: Path
    destination: Path
    source_sha256: str
    destination_sha256: str
    file_count: int
    byte_count: int
    dry_run: bool


@dataclass(frozen=True)
class _CopySnapshot:
    metadata: os.stat_result
    sha256: str
    size: int
    sample_sha256: str


def _absolute(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = Path.cwd() / value
    return Path(os.path.normpath(value))


def _safe_relative(value: str | os.PathLike[str], field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be relative and traversal-free")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} must be relative and traversal-free")
    return path


def _safe_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or ".." in value:
        raise ValueError(f"{field} must be a safe name")
    return value


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlink is forbidden in artifact path: {current}")


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


def _regular_metadata(path: Path, *, label: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink is forbidden in {label}: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must contain only regular files: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"hardlink is forbidden in {label}: {path}")
    return metadata


def _hash_fd(fd: int, chunk_size: int = _CHUNK_SIZE) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sample_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    width = min(64 * 1024, size)
    offsets = sorted({0, max(0, (size - width) // 2), max(0, size - width)})
    for offset in offsets:
        digest.update(os.pread(fd, width, offset))
    return digest.hexdigest()


def _sample_path(path: Path, size: int) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        return _sample_fd(fd, size)
    finally:
        os.close(fd)


def _stable_hash_file(path: Path) -> _CopySnapshot:
    before_path = _regular_metadata(path, label="legacy source")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before_fd = os.fstat(fd)
        if not _same_file(before_path, before_fd):
            raise ValueError(f"legacy source changed before hashing: {path}")
        before_sample = _sample_fd(fd, before_fd.st_size)
        digest, size = _hash_fd(fd)
        after_fd = os.fstat(fd)
        after_sample = _sample_fd(fd, after_fd.st_size)
    finally:
        os.close(fd)
    after_path = _regular_metadata(path, label="legacy source")
    if (
        not _same_file(before_fd, after_fd)
        or not _same_file(before_fd, after_path)
        or before_sample != after_sample
    ):
        raise ValueError(f"legacy source changed while hashing: {path}")
    return _CopySnapshot(after_path, digest, size, after_sample)


def _copy_open_file(source: Path, temporary: Path, chunk_size: int = _CHUNK_SIZE) -> _CopySnapshot:
    before_path = _regular_metadata(source, label="legacy source")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    digest = hashlib.sha256()
    size = 0
    try:
        before_fd = os.fstat(source_fd)
        if not _same_file(before_path, before_fd):
            raise ValueError(f"legacy source changed before copying: {source}")
        before_sample = _sample_fd(source_fd, before_fd.st_size)
        while True:
            chunk = os.read(source_fd, chunk_size)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
            digest.update(chunk)
            size += len(chunk)
        os.fsync(destination_fd)
        after_fd = os.fstat(source_fd)
        after_sample = _sample_fd(source_fd, after_fd.st_size)
    finally:
        os.close(destination_fd)
        os.close(source_fd)
    after_path = _regular_metadata(source, label="legacy source")
    if (
        not _same_file(before_fd, after_fd)
        or not _same_file(before_fd, after_path)
        or before_sample != after_sample
    ):
        raise ValueError(f"legacy source changed while copying: {source}")
    return _CopySnapshot(after_path, digest.hexdigest(), size, after_sample)


def _ensure_safe_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    _assert_no_symlink_components(current)
    if not stat.S_ISDIR(current.lstat().st_mode):
        raise ValueError(f"artifact parent is not a directory: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"unsafe artifact directory: {directory}")


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        # Atomic no-clobber fallback for non-glibc Unix; the temporary link is
        # removed before the destination is accepted as a private regular file.
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
    else:
        renameat2.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100, os.fsencode(temporary), -100, os.fsencode(destination), 1
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), destination)
            raise OSError(error, os.strerror(error), destination)
    metadata = destination.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"atomic publication produced an unsafe destination: {destination}")
    _fsync_directory(destination.parent)


def _walk_source(source: Path) -> tuple[bool, list[tuple[Path, Path]]]:
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink is forbidden in legacy source: {source}")
    if stat.S_ISREG(metadata.st_mode):
        _regular_metadata(source, label="legacy source")
        return False, [(source, Path(source.name))]
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"legacy source must be a regular file or directory: {source}")

    files: list[tuple[Path, Path]] = []

    def visit(directory: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            entry_metadata = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            if stat.S_ISLNK(entry_metadata.st_mode):
                raise ValueError(f"symlink is forbidden in legacy source: {path}")
            if stat.S_ISDIR(entry_metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(entry_metadata.st_mode):
                if entry_metadata.st_nlink != 1:
                    raise ValueError(f"hardlink is forbidden in legacy source: {path}")
                files.append((path, path.relative_to(source)))
            else:
                raise ValueError(f"special file is forbidden in legacy source: {path}")

    visit(source)
    return True, files


def _tree_sha256(entries: Iterable[tuple[Path, str, int]]) -> str:
    digest = hashlib.sha256()
    for relative, file_digest, size in entries:
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _copy_or_verify(source: Path, destination: Path, *, dry_run: bool) -> tuple[str, str, int]:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError(f"symlink is forbidden in artifact destination: {destination}")
        source_snapshot = _stable_hash_file(source)
        try:
            destination_snapshot = _stable_hash_file(destination)
        except ValueError as exc:
            raise FileExistsError(f"destination collision: {destination}") from exc
        if (
            source_snapshot.sha256 != destination_snapshot.sha256
            or source_snapshot.size != destination_snapshot.size
        ):
            raise FileExistsError(f"destination collision: {destination}")
        return source_snapshot.sha256, destination_snapshot.sha256, source_snapshot.size

    if dry_run:
        snapshot = _stable_hash_file(source)
        return snapshot.sha256, snapshot.sha256, snapshot.size

    _ensure_safe_directory(destination.parent)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    try:
        snapshot = _copy_open_file(source, temporary)
        after_copy = _regular_metadata(source, label="legacy source")
        if (
            not _same_file(snapshot.metadata, after_copy)
            or _sample_path(source, after_copy.st_size) != snapshot.sample_sha256
        ):
            raise ValueError(f"legacy source changed while copying: {source}")
        try:
            _publish_no_replace(temporary, destination)
        except FileExistsError:
            existing = _stable_hash_file(destination)
            if existing.sha256 != snapshot.sha256 or existing.size != snapshot.size:
                raise FileExistsError(f"destination collision: {destination}")
            temporary.unlink(missing_ok=True)
        verified = _stable_hash_file(destination)
        if verified.sha256 != snapshot.sha256 or verified.size != snapshot.size:
            raise ValueError(f"destination hash verification failed: {destination}")
        final_source = _regular_metadata(source, label="legacy source")
        if not _same_file(snapshot.metadata, final_source):
            raise ValueError(f"legacy source changed while copying: {source}")
        return snapshot.sha256, verified.sha256, snapshot.size
    finally:
        temporary.unlink(missing_ok=True)


def migrate_critical_run(
    source: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    *,
    move: bool = False,
    destination: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> MigrationResult:
    if move:
        raise ValueError("legacy migration has no move or delete operation")
    source_path = _absolute(source)
    artifact_path = _absolute(artifact_root)
    _assert_no_symlink_components(source_path)
    _assert_no_symlink_components(artifact_path)
    is_directory, files = _walk_source(source_path)
    target_relative = _safe_relative(destination or source_path.name, "destination path")
    target = artifact_path / target_relative
    if target != artifact_path / target.relative_to(artifact_path):
        raise ValueError("destination path must be relative and traversal-free")

    if not dry_run:
        _ensure_safe_directory(artifact_path)
        if is_directory:
            if target.exists() and not target.is_dir():
                raise FileExistsError(f"destination collision: {target}")
            _ensure_safe_directory(target)
        elif target.exists() and target.is_dir():
            raise FileExistsError(f"destination collision: {target}")

    source_entries: list[tuple[Path, str, int]] = []
    destination_entries: list[tuple[Path, str, int]] = []
    for path, relative in files:
        destination_file = target / relative if is_directory else target
        source_digest, destination_digest, size = _copy_or_verify(
            path, destination_file, dry_run=dry_run
        )
        tree_relative = relative if is_directory else Path(source_path.name)
        source_entries.append((tree_relative, source_digest, size))
        destination_entries.append((tree_relative, destination_digest, size))

    if is_directory:
        _, after_files = _walk_source(source_path)
        if [relative for _, relative in after_files] != [relative for _, relative in files]:
            raise ValueError(f"legacy source changed while copying: {source_path}")
        if not dry_run:
            _, destination_files = _walk_source(target)
            if [relative for _, relative in destination_files] != [
                relative for _, relative in files
            ]:
                raise FileExistsError(f"destination collision: {target}")
    source_tree = _tree_sha256(source_entries)
    destination_tree = _tree_sha256(destination_entries)
    if source_tree != destination_tree:
        raise ValueError("destination tree hash verification failed")
    return MigrationResult(
        source=source_path,
        destination=target,
        source_sha256=source_tree,
        destination_sha256=destination_tree,
        file_count=len(files),
        byte_count=sum(size for _, _, size in source_entries),
        dry_run=dry_run,
    )


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) >= 1024 * 1024:
        raise ValueError("migration JSON must remain below 1 MiB")
    return encoded


def _publish_json_equal_or_fail(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _json_bytes(payload)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _regular_metadata(path, label="artifact destination").st_size != len(encoded):
            raise FileExistsError(f"destination collision: {path}")
        if path.read_bytes() != encoded:
            raise FileExistsError(f"destination collision: {path}")
        return
    _ensure_safe_directory(path.parent)
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _publish_no_replace(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(f"destination collision: {path}")
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def _load_mapping(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "runs"}:
        raise ValueError("critical run mapping must contain schema_version and runs")
    if payload["schema_version"] != 1 or not isinstance(payload["runs"], list):
        raise ValueError("unsupported critical run mapping")
    return payload["runs"]


def migrate_mapping(
    mapping: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] = ".",
    dry_run: bool = False,
    verify: bool = True,
) -> dict[str, object]:
    if not verify:
        raise ValueError("critical migration cannot disable hash verification")
    mapping_path = _absolute(mapping)
    root = _absolute(source_root)
    artifact_path = _absolute(artifact_root)
    _assert_no_symlink_components(mapping_path)
    _assert_no_symlink_components(root)
    runs = _load_mapping(mapping_path)
    unresolved: list[str] = []
    migrated: list[dict[str, object]] = []
    total_files = 0
    total_bytes = 0

    for raw_run in runs:
        if not isinstance(raw_run, dict):
            raise ValueError("critical run entry must be an object")
        name = _safe_name(raw_run.get("name"), "run name")
        resolution = raw_run.get("resolution")
        classification = raw_run.get("classification")
        sources = raw_run.get("sources")
        if resolution == "unresolved":
            if classification != "unresolved" or sources != [] or raw_run.get("run_id") is not None:
                raise ValueError(f"unresolved mapping is inconsistent: {name}")
            unresolved.append(name)
            continue
        if resolution != "resolved" or classification != "critical":
            raise ValueError(f"resolved mapping must be classified critical: {name}")
        dataset = _safe_name(raw_run.get("dataset"), "dataset")
        stage = raw_run.get("stage")
        if stage not in _ALLOWED_STAGES:
            raise ValueError(f"invalid migration stage: {stage}")
        run_id = _safe_name(raw_run.get("run_id"), "run_id")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"resolved mapping has no sources: {name}")
        run_root_relative = Path(dataset) / str(stage) / run_id
        run_root = artifact_path / run_root_relative
        results: list[MigrationResult] = []
        source_descriptors: list[dict[str, object]] = []
        for raw_source in sources:
            if not isinstance(raw_source, dict) or set(raw_source) != {"path", "destination"}:
                raise ValueError(f"invalid source mapping for {name}")
            source_relative = _safe_relative(raw_source["path"], "source path")
            destination_relative = _safe_relative(raw_source["destination"], "destination path")
            source_path = root / source_relative
            result = migrate_critical_run(
                source_path,
                run_root,
                destination=destination_relative,
                dry_run=dry_run,
            )
            results.append(result)
            source_descriptors.append(
                {
                    "path": source_relative.as_posix(),
                    "destination": destination_relative.as_posix(),
                    "sha256": result.source_sha256,
                    "files": result.file_count,
                    "bytes": result.byte_count,
                }
            )
        aggregate_source = hashlib.sha256(
            "".join(result.source_sha256 for result in results).encode("ascii")
        ).hexdigest()
        aggregate_destination = hashlib.sha256(
            "".join(result.destination_sha256 for result in results).encode("ascii")
        ).hexdigest()
        if aggregate_source != aggregate_destination:
            raise ValueError(f"run migration hash verification failed: {name}")
        provenance: dict[str, object] = {
            "schema_version": 1,
            "name": name,
            "dataset": dataset,
            "stage": stage,
            "run_id": run_id,
            "classification": "critical",
            "operation": "copy-plus-verify",
            "source_sha256": aggregate_source,
            "destination_sha256": aggregate_destination,
            "file_count": sum(result.file_count for result in results),
            "byte_count": sum(result.byte_count for result in results),
            "sources": source_descriptors,
        }
        if not dry_run:
            _publish_json_equal_or_fail(
                run_root / "provenance" / "legacy-migration.json", provenance
            )
        migrated.append(provenance)
        total_files += int(provenance["file_count"])
        total_bytes += int(provenance["byte_count"])

    return {
        "schema_version": 1,
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "operation": "dry-run" if dry_run else "copy-plus-verify",
        "resolved_runs": len(migrated),
        "unresolved_runs": unresolved,
        "file_count": total_files,
        "byte_count": total_bytes,
        "runs": migrated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.copy and not args.dry_run:
        raise ValueError("migration requires --copy (move/delete is unsupported)")
    if not args.verify:
        raise ValueError("migration requires --verify")
    report = migrate_mapping(
        args.mapping,
        args.artifact_root,
        source_root=args.source_root,
        dry_run=args.dry_run,
        verify=True,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
