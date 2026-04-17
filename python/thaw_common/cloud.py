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

Future (Rust/thaw-cloud crate): concurrent ranged GETs directly into
pinned host memory with overlapping DMA. This Python layer is ~1x S3
bandwidth; the Rust layer should saturate NIC (10+ Gbps).
"""

import hashlib
import os
from typing import Optional
from urllib.parse import urlparse


DEFAULT_CACHE_DIR = os.environ.get(
    "THAW_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "thaw", "snapshots"),
)

_REMOTE_SCHEMES = {"s3", "gs", "http", "https"}


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
) -> str:
    """Resolve a URI to a local path. Downloads remote URIs to cache.

    Local paths (including None/empty) pass through unchanged, so this
    is safe to call unconditionally at the top of any restore path.
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
        _download_s3(uri, tmp)
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


def _s3_client():
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "thaw cloud S3 support requires boto3. "
            "Install with: pip install thaw-vllm[cloud]"
        ) from e
    return boto3.client("s3")


def _download_s3(uri: str, dest: str) -> None:
    bucket, key = _parse_s3(uri)
    client = _s3_client()
    try:
        client.download_file(bucket, key, dest)
    except Exception as e:
        from botocore.exceptions import ClientError, NoCredentialsError
        if isinstance(e, NoCredentialsError):
            raise RuntimeError(
                f"thaw S3 download failed ({uri}): no AWS credentials found. "
                f"Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, use `aws configure`, "
                f"or attach an IAM role."
            ) from e
        if isinstance(e, ClientError):
            code = e.response.get("Error", {}).get("Code", "?")
            if code in ("404", "NoSuchKey", "NoSuchBucket"):
                raise FileNotFoundError(
                    f"thaw S3 download failed: object not found at {uri}"
                ) from e
            if code in ("403", "AccessDenied"):
                raise PermissionError(
                    f"thaw S3 download failed: access denied for {uri}. "
                    f"Check IAM permissions for s3:GetObject."
                ) from e
            raise RuntimeError(
                f"thaw S3 download failed ({uri}): {code} — {e}"
            ) from e
        raise


def _upload_s3(local_path: str, uri: str) -> None:
    bucket, key = _parse_s3(uri)
    client = _s3_client()
    try:
        client.upload_file(local_path, bucket, key)
    except Exception as e:
        from botocore.exceptions import ClientError, NoCredentialsError
        if isinstance(e, NoCredentialsError):
            raise RuntimeError(
                f"thaw S3 upload failed ({uri}): no AWS credentials found. "
                f"Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, use `aws configure`, "
                f"or attach an IAM role."
            ) from e
        if isinstance(e, ClientError):
            code = e.response.get("Error", {}).get("Code", "?")
            if code in ("403", "AccessDenied"):
                raise PermissionError(
                    f"thaw S3 upload failed: access denied for {uri}. "
                    f"Check IAM permissions for s3:PutObject."
                ) from e
            raise RuntimeError(
                f"thaw S3 upload failed ({uri}): {code} — {e}"
            ) from e
        raise
