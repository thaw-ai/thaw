"""
thaw_common.cloud — resolve remote snapshot URIs to local paths.

MVP: download-on-demand S3 support. Given an s3://bucket/key URI,
download to a local cache dir and return the local path. Works
transparently with all thaw entry points — the hook is at the loader
layer, before any filesystem operations.

Cache behavior:
  - Default location: ~/.cache/thaw/snapshots (override via THAW_CACHE_DIR)
  - Stable path per-URI via sha256 hash prefix, so repeat loads hit cache
  - Atomic via .part rename — a crashed download doesn't poison the cache

Throughput:
  - Parallel ranged-GET with a shared boto3 client + ThreadPoolExecutor.
    Writes go through os.pwrite into a preallocated file, so ranges land
    at the right offsets without coordination.
  - Tunables: THAW_S3_CONCURRENCY (default 32), THAW_S3_PART_SIZE_MB
    (default 16), THAW_S3_MULTIPART_THRESHOLD_MB (default 16).
  - Target: saturate NIC (10+ Gbps) from a single process. Measured
    >800 MB/s on 10 Gbps EC2 egress; boto3.download_file caps ~67 MB/s.

Future (Rust/thaw-cloud crate): ranged GETs directly into WC-pinned host
memory with overlapping CUDA DMA, skipping the local file entirely.
"""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
from urllib.parse import urlparse


DEFAULT_CACHE_DIR = os.environ.get(
    "THAW_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "thaw", "snapshots"),
)

_REMOTE_SCHEMES = {"s3", "gs", "http", "https"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


# Default tuning: 32 parallel GETs × 16 MiB parts = 512 MiB in flight.
# Works well on 10 Gbps links; scale up for 25/100 Gbps pods.
_DEFAULT_CONCURRENCY = _env_int("THAW_S3_CONCURRENCY", 32)
_DEFAULT_PART_SIZE = _env_int("THAW_S3_PART_SIZE_MB", 16) * 1024 * 1024
_DEFAULT_MULTIPART_THRESHOLD = _env_int("THAW_S3_MULTIPART_THRESHOLD_MB", 16) * 1024 * 1024
_DEFAULT_READ_CHUNK = 4 * 1024 * 1024  # how much body.read() returns per call


def is_remote(uri: str) -> bool:
    """True if uri is a supported remote URI (s3://, gs://, http(s)://)."""
    if not uri or "://" not in uri:
        return False
    scheme = uri.split("://", 1)[0].lower()
    return scheme in _REMOTE_SCHEMES


def _cache_path(uri: str, cache_dir: str) -> str:
    parsed = urlparse(uri)
    basename = os.path.basename(parsed.path) or "snapshot.thaw"
    digest = hashlib.sha256(uri.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, digest, basename)


def resolve_snapshot_path(
    uri: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Resolve a URI to a local path. Downloads remote URIs to cache.

    Local paths (including None/empty) pass through unchanged, so this
    is safe to call unconditionally at the top of any restore path.

    `progress(bytes_done, bytes_total)` fires periodically during S3
    downloads. Omit for silent mode.
    """
    if not uri or not is_remote(uri):
        return uri

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    local = _cache_path(uri, cache_dir)

    if os.path.exists(local) and not force:
        return local

    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".part"

    scheme = urlparse(uri).scheme.lower()
    if scheme == "s3":
        _download_s3(uri, tmp, progress=progress)
    else:
        raise NotImplementedError(
            f"thaw cloud: scheme '{scheme}' not yet supported. "
            f"Currently supported: s3://"
        )

    os.rename(tmp, local)
    return local


def upload_snapshot(local_path: str, uri: str) -> None:
    """Upload a local .thaw file to a remote URI.

    For TP, call once per rank file — use rank_snapshot_path to generate
    per-rank S3 keys the same way local per-rank paths are generated.
    """
    if not is_remote(uri):
        raise ValueError(
            f"upload_snapshot destination must be a remote URI, got: {uri}"
        )
    scheme = urlparse(uri).scheme.lower()
    if scheme == "s3":
        _upload_s3(local_path, uri)
    else:
        raise NotImplementedError(
            f"thaw cloud: upload to '{scheme}' not yet supported. "
            f"Currently supported: s3://"
        )


def _parse_s3(uri: str):
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri} (expected s3://bucket/key)")
    return bucket, key


def _s3_client(concurrency: int = _DEFAULT_CONCURRENCY):
    """Return a boto3 S3 client with a connection pool sized for `concurrency`.

    botocore's default pool is 10; undersized pools serialize concurrent GETs.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError(
            "thaw cloud S3 support requires boto3. "
            "Install with: pip install thaw-vllm[cloud]"
        ) from e
    cfg = Config(
        max_pool_connections=max(concurrency + 4, 16),
        retries={"max_attempts": 5, "mode": "adaptive"},
        tcp_keepalive=True,
    )
    return boto3.client("s3", config=cfg)


def _map_boto_error(uri: str, op: str, e: BaseException) -> BaseException:
    """Map boto3 errors to friendly Python exception types.

    Introspects by class name + `.response` attribute so this works even when
    botocore isn't importable in the current thread context.
    """
    cls = type(e).__name__
    if cls == "NoCredentialsError":
        return RuntimeError(
            f"thaw S3 {op} failed ({uri}): no AWS credentials found. "
            f"Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, use `aws configure`, "
            f"or attach an IAM role."
        )
    response = getattr(e, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "?")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", None)
        if code in ("404", "NoSuchKey", "NoSuchBucket") or status == 404:
            return FileNotFoundError(
                f"thaw S3 {op} failed: object not found at {uri}"
            )
        if code in ("403", "AccessDenied") or status == 403:
            return PermissionError(
                f"thaw S3 {op} failed: access denied for {uri}. "
                f"Check IAM permissions."
            )
        return RuntimeError(f"thaw S3 {op} failed ({uri}): {code} — {e}")
    return e


def _download_s3(
    uri: str,
    dest: str,
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
    part_size: int = _DEFAULT_PART_SIZE,
    multipart_threshold: int = _DEFAULT_MULTIPART_THRESHOLD,
    progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Parallel ranged-GET download. Preallocates dest and scatters writes via pwrite.

    Falls back to a single GetObject for small files (below multipart_threshold)
    to avoid HEAD + executor overhead.
    """
    bucket, key = _parse_s3(uri)
    client = _s3_client(concurrency=concurrency)

    # Discover size. HEAD also validates existence and perms before we touch disk.
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise _map_boto_error(uri, "download", e) from e
    size = int(head["ContentLength"])

    if size == 0:
        open(dest, "wb").close()
        if progress:
            progress(0, 0)
        return

    # Preallocate: truncate gives us the full file length so pwrite into any
    # range lands at the right offset without extending.
    with open(dest, "wb") as f:
        f.truncate(size)

    # Small-file fast path: single GET, no executor.
    if size <= multipart_threshold:
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
            fd = os.open(dest, os.O_WRONLY)
            try:
                _stream_body_to_fd(resp["Body"], fd, 0, progress, size)
            finally:
                os.close(fd)
        except Exception as e:
            raise _map_boto_error(uri, "download", e) from e
        return

    ranges = []
    start = 0
    while start < size:
        end = min(start + part_size, size) - 1
        ranges.append((start, end))
        start = end + 1

    # Shared counter for progress. GIL makes int += atomic enough for this.
    state = {"done": 0}
    total = size

    def _emit_progress(delta: int):
        state["done"] += delta
        if progress:
            progress(state["done"], total)

    fd = os.open(dest, os.O_WRONLY)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(_download_range, client, bucket, key, fd, s, e, _emit_progress)
                for (s, e) in ranges
            ]
            first_error: Optional[BaseException] = None
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as inner:
                    if first_error is None:
                        first_error = inner
            if first_error is not None:
                raise _map_boto_error(uri, "download", first_error) from first_error
    finally:
        os.close(fd)


def _download_range(
    client,
    bucket: str,
    key: str,
    fd: int,
    start: int,
    end: int,
    emit_progress: Callable[[int], None],
    *,
    retries: int = 4,
) -> None:
    """GET bytes[start..end] and pwrite into fd at `start`. Retries on transient errors."""
    attempt = 0
    while True:
        try:
            resp = client.get_object(
                Bucket=bucket, Key=key, Range=f"bytes={start}-{end}"
            )
            _stream_body_to_fd(resp["Body"], fd, start, lambda n: emit_progress(n), None)
            return
        except Exception as e:
            # Introspect via attribute instead of importing botocore (thread-safe fallback).
            response = getattr(e, "response", None)
            code = (
                response.get("Error", {}).get("Code", "?")
                if isinstance(response, dict)
                else None
            )
            if code in ("403", "AccessDenied", "404", "NoSuchKey", "NoSuchBucket"):
                raise  # terminal — don't waste retries
            if attempt >= retries:
                raise
        attempt += 1
        time.sleep(min(0.1 * (2 ** attempt), 2.0))


def _stream_body_to_fd(
    body,
    fd: int,
    offset: int,
    progress: Optional[Callable[[int], None]],
    _total: Optional[int],
) -> None:
    """Read streaming body in chunks, pwrite at (offset + cursor).

    `progress(delta)` fires per write — receives bytes-this-chunk so callers can
    accumulate. On callers that don't need it, pass a no-op.
    """
    cursor = 0
    while True:
        chunk = body.read(_DEFAULT_READ_CHUNK)
        if not chunk:
            break
        n = 0
        view = memoryview(chunk)
        while n < len(view):
            written = os.pwrite(fd, view[n:], offset + cursor + n)
            if written <= 0:
                raise OSError("pwrite returned 0")
            n += written
        cursor += len(chunk)
        if progress is not None:
            progress(len(chunk))


def _upload_s3(local_path: str, uri: str) -> None:
    """Parallel multipart upload via boto3 TransferConfig."""
    bucket, key = _parse_s3(uri)
    client = _s3_client()
    try:
        from boto3.s3.transfer import TransferConfig
        cfg = TransferConfig(
            multipart_threshold=_DEFAULT_MULTIPART_THRESHOLD,
            multipart_chunksize=_DEFAULT_PART_SIZE,
            max_concurrency=_DEFAULT_CONCURRENCY,
            use_threads=True,
        )
        client.upload_file(local_path, bucket, key, Config=cfg)
    except Exception as e:
        raise _map_boto_error(uri, "upload", e) from e
