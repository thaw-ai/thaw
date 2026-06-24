"""
thaw_common.cloud — resolve remote snapshot URIs to local paths.

MVP: download-on-demand S3 support. Given an s3://bucket/key URI,
download to a local cache dir and return the local path. Works
transparently with all thaw entry points — the hook is at the loader
layer, before any filesystem operations.

For a corpus of many objects (RAG shards, per-rank files), resolve_snapshots
and upload_snapshots transfer them with bounded cross-file concurrency.

Cache behavior:
  - Default location: ~/.cache/thaw/snapshots (override via THAW_CACHE_DIR)
  - Stable path per-URI via sha256 hash prefix, so repeat loads hit cache
  - Atomic via .part rename — a crashed download doesn't poison the cache

Throughput:
  - Single object: parallel ranged-GET with a shared boto3 client +
    ThreadPoolExecutor. Writes go through os.pwrite into a preallocated file,
    so ranges land at the right offsets without coordination. Note that a
    single S3 key is server-throttled to ~135 MB/s regardless of how many
    concurrent ranges you issue (flat 8→256 conc., measured EC2 intra-region;
    see docs/ARCHITECTURE.md). Ranged concurrency mainly hides per-request
    latency and TLS/connection warm-up, not raw single-key bandwidth.
  - Many objects: the real bandwidth lever is fanning out across DISTINCT
    keys — each key has its own ~135 MB/s budget, so N keys in parallel give
    ~N×135 MB/s. resolve_snapshots/upload_snapshots provide that fan-out; it
    is also the read side of the "shard at freeze time" plan in ARCHITECTURE.
  - Client reuse: the boto3 client is built once per process and cached
    (see _s3_client), so repeated transfers keep credentials resolved and
    TCP/TLS connections warm instead of paying a fresh handshake each time.
  - Tunables: THAW_S3_CONCURRENCY (per-file ranged GETs, default 32),
    THAW_S3_FILE_CONCURRENCY (cross-file fan-out, default 8),
    THAW_S3_PART_SIZE_MB (default 16), THAW_S3_MULTIPART_THRESHOLD_MB
    (default 16).

Future (Rust/thaw-cloud crate): ranged GETs directly into WC-pinned host
memory with overlapping CUDA DMA, skipping the local file entirely.
"""

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Child of the "thaw" logger configured in thaw_common.telemetry. Using a
# named child here (rather than importing telemetry's logger) keeps cloud.py
# import-light — benchmarks/bench_s3_download.py loads this module standalone,
# bypassing the package __init__. Mirrors the "thaw.pool" logger in cli.py.
logger = logging.getLogger("thaw.cloud")


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

# Cross-file concurrency for the batch helpers (resolve_snapshots /
# upload_snapshots). This bounds how many INDEPENDENT objects transfer in
# parallel — the win for a many-object corpus (RAG chunks, per-rank shards,
# weights + KV sidecar) where each object is small enough that per-file
# ranged concurrency doesn't help. Total concurrent S3 requests are still
# capped by the shared client's connection pool (see _s3_client), so this is
# bounded, not unbounded fan-out. Default 8: enough to hide per-object
# request latency without risking per-prefix throttling.
_DEFAULT_FILE_CONCURRENCY = _env_int("THAW_S3_FILE_CONCURRENCY", 8)


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


def _batch_workers(n_items: int, max_files: Optional[int]) -> int:
    """How many files to transfer in parallel: bounded by item count and limit."""
    limit = max_files if max_files is not None else _DEFAULT_FILE_CONCURRENCY
    return max(1, min(n_items, limit))


def resolve_snapshots(
    uris: Sequence[str],
    cache_dir: Optional[str] = None,
    force: bool = False,
    max_files: Optional[int] = None,
) -> List[str]:
    """Resolve many URIs to local paths, downloading remote objects in parallel.

    This is the many-object analogue of resolve_snapshot_path. Per-URI
    behavior is IDENTICAL to calling resolve_snapshot_path on each one: local
    paths (and None/empty) pass through unchanged, remote URIs download into
    the cache with the same content-addressed layout, the same atomic .part
    rename, the same cache-hit skip, and the same typed errors.

    The difference is that independent remote objects download concurrently
    with a bounded thread pool (THAW_S3_FILE_CONCURRENCY, default 8, override
    per-call with `max_files`), reusing the shared cached S3 client. A corpus
    of N objects then costs ~max(per-object latency) instead of ~sum, while
    total in-flight S3 requests stay bounded by the client connection pool.

    Returns local paths in the SAME ORDER as `uris`. If any download fails,
    the first error is re-raised after the other in-flight transfers settle;
    because each file commits via an atomic rename, a failed or cancelled
    download never leaves a partial object in the cache.

    Note: for a batch of many LARGE objects, per-file ranged concurrency
    (THAW_S3_CONCURRENCY) and file concurrency multiply; the shared client's
    connection pool is the hard global bound. Tune both knobs to your NIC.
    """
    uris = list(uris)
    results: List[Optional[str]] = [None] * len(uris)

    # Local paths and empties resolve instantly — no thread, no client.
    remote_idx = []
    for i, uri in enumerate(uris):
        if not uri or not is_remote(uri):
            results[i] = uri
        else:
            remote_idx.append(i)

    if not remote_idx:
        return results  # type: ignore[return-value]

    # De-duplicate by exact URI before fanning out. Identical URIs resolve to
    # the SAME content-addressed cache file, so downloading them on separate
    # threads would race on the shared `<path>.part` temp file and its rename
    # (torn cache on POSIX; a spurious FileExistsError on Windows, where rename
    # won't replace an existing target). Download each DISTINCT URI once and
    # fan the resolved path out to every position that requested it.
    uri_to_indices = {}
    distinct_uris: List[str] = []
    for i in remote_idx:
        u = uris[i]
        if u not in uri_to_indices:
            uri_to_indices[u] = []
            distinct_uris.append(u)
        uri_to_indices[u].append(i)

    workers = _batch_workers(len(distinct_uris), max_files)
    logger.debug(
        "resolve_snapshots: %d remote objects (%d distinct), %d-way file concurrency",
        len(remote_idx), len(distinct_uris), workers,
    )

    first_error: Optional[BaseException] = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_uri = {
            ex.submit(
                resolve_snapshot_path, u, cache_dir=cache_dir, force=force
            ): u
            for u in distinct_uris
        }
        for fut in as_completed(fut_to_uri):
            u = fut_to_uri[fut]
            try:
                path = fut.result()
                for i in uri_to_indices[u]:
                    results[i] = path
            except BaseException as e:  # noqa: BLE001 — capture first, surface below
                if first_error is None:
                    first_error = e
                else:
                    # Don't lose later failures silently; the first is raised.
                    logger.debug(
                        "resolve_snapshots: additional failure for %s: %r", u, e
                    )
    if first_error is not None:
        raise first_error
    return results  # type: ignore[return-value]


def upload_snapshots(
    pairs: Sequence[Tuple[str, str]],
    max_files: Optional[int] = None,
) -> None:
    """Upload many (local_path, uri) pairs in parallel.

    The many-object analogue of upload_snapshot. Per-pair behavior is
    identical (each goes through the same multipart upload + typed errors);
    independent uploads run concurrently with a bounded thread pool
    (THAW_S3_FILE_CONCURRENCY, override with `max_files`) reusing the shared
    client. The first error is re-raised after in-flight uploads settle.

    Uploads are independent objects, so a partial-batch failure leaves the
    objects that did succeed in place (S3 PUTs are atomic per object) and
    raises for the rest — no object is left half-written.
    """
    pairs = list(pairs)
    if not pairs:
        return

    workers = _batch_workers(len(pairs), max_files)
    logger.debug("upload_snapshots: %d objects, %d-way concurrency", len(pairs), workers)

    first_error: Optional[BaseException] = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(upload_snapshot, local_path, uri)
            for (local_path, uri) in pairs
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as e:  # noqa: BLE001 — capture first, surface below
                if first_error is None:
                    first_error = e
                else:
                    logger.debug("upload_snapshots: additional upload failure: %r", e)
    if first_error is not None:
        raise first_error


def _parse_s3(uri: str):
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri} (expected s3://bucket/key)")
    return bucket, key


# --- shared client cache --------------------------------------------------
#
# Building a boto3 client is not free: it resolves the credential provider
# chain (which can issue an IMDS network round-trip on an EC2/pod role),
# loads the S3 service model, and resolves the endpoint — ~3 ms warm and
# hundreds of ms cold. More importantly, a *fresh* client starts with an
# *empty* connection pool, so the first request on it pays a new TCP+TLS
# handshake to S3 instead of reusing a warm keep-alive connection. The
# previous implementation built a new client on every download and every
# upload, throwing that warmth away each time.
#
# We cache one client per process, keyed by (pid, pool_size):
#   - pid: botocore clients are NOT fork-safe. vLLM spawns/forks TP worker
#     processes; a client (and its live sockets) built in the parent must
#     never be reused in a child. Re-keying on os.getpid() transparently
#     rebuilds the client after a fork, so a forked worker gets its own.
#   - pool_size: callers that want a bigger HTTP connection pool get a
#     distinctly-keyed client sized for their concurrency.
#
# The low-level boto3 client is documented as thread-safe for concurrent
# calls, so a single cached client is safe to share across the ranged-GET
# ThreadPoolExecutor (which the previous code already relied on) and across
# the batch helpers. The cache dict itself is guarded by a lock.
_client_cache = {}
_client_cache_lock = threading.Lock()


def _build_s3_client(pool_size: int):
    """Construct a fresh boto3 S3 client with a connection pool of `pool_size`."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError(
            "thaw cloud S3 support requires boto3. "
            "Install with: pip install thaw-vllm[cloud]"
        ) from e
    cfg = Config(
        max_pool_connections=pool_size,
        retries={"max_attempts": 5, "mode": "adaptive"},
        tcp_keepalive=True,
    )
    return boto3.client("s3", config=cfg)


def _s3_client(concurrency: int = _DEFAULT_CONCURRENCY):
    """Return a process-cached boto3 S3 client with a pool sized for `concurrency`.

    botocore's default pool is 10; undersized pools serialize concurrent GETs,
    so we size it to ``concurrency + 4`` (minimum 16) — identical to the pool
    the previous per-call implementation used. The client is then cached and
    reused so repeated downloads/uploads keep credentials resolved and TCP/TLS
    connections warm. See the module-level note for the (pid, pool_size) key
    and fork/thread-safety rationale.
    """
    pool_size = max(concurrency + 4, 16)
    key = (os.getpid(), pool_size)
    # Fast path: lock-free read of an already-built client.
    client = _client_cache.get(key)
    if client is not None:
        return client
    # Slow path: build once under the lock (double-checked so concurrent
    # first-callers don't each build a client).
    with _client_cache_lock:
        client = _client_cache.get(key)
        if client is None:
            client = _build_s3_client(pool_size)
            _client_cache[key] = client
        return client


def reset_s3_client_cache() -> None:
    """Drop all cached S3 clients.

    The normal runtime never needs this — the (pid, pool_size) key already
    rebuilds the client after a fork. It exists so tests that swap AWS
    credentials/endpoints between cases can force a clean rebuild, and so
    operators can recover from a wedged client without restarting the process.
    """
    with _client_cache_lock:
        _client_cache.clear()


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
