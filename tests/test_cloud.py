"""
Tests for thaw_common.cloud — S3 URI resolution and cache behavior.

Two layers of coverage:
  - MagicMock-patched tests exercise the dispatch + parallel ranged-GET
    bookkeeping without boto3
  - moto `@mock_aws` tests exercise the actual boto3 code path against
    an in-process fake S3 server, so bucket/key parsing, Range requests,
    and error mapping are validated for real.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

try:
    import boto3
    from moto import mock_aws
    HAVE_MOTO = True
except ImportError:
    HAVE_MOTO = False
    # no-op shim so class-body @mock_aws decorators parse cleanly when moto is absent.
    def mock_aws(fn):
        return fn

try:
    import botocore  # noqa: F401
    HAVE_BOTO = True
except ImportError:
    HAVE_BOTO = False

# Ensure python/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from thaw_common.cloud import (
    is_remote,
    resolve_snapshot_path,
    resolve_snapshots,
    upload_snapshot,
    upload_snapshots,
    reset_s3_client_cache,
    _cache_path,
    _parse_s3,
    _s3_client,
)
import thaw_common.cloud as cloud


def _mock_s3_with_payload(payload: bytes) -> MagicMock:
    """Build a MagicMock S3 client that serves `payload` via head_object + get_object."""
    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {"ContentLength": len(payload)}

    def fake_get_object(Bucket, Key, Range=None):
        if Range is None:
            data = payload
        else:
            # Range header: "bytes=<start>-<end>"
            spec = Range.split("=", 1)[1]
            start_s, end_s = spec.split("-", 1)
            start, end = int(start_s), int(end_s)
            data = payload[start:end + 1]
        body = MagicMock()
        # Return all bytes on the first read, then EOF.
        body.read.side_effect = [data, b""]
        return {"Body": body, "ContentLength": len(data)}

    mock_s3.get_object.side_effect = fake_get_object
    return mock_s3


class TestIsRemote(unittest.TestCase):
    def test_local_paths(self):
        self.assertFalse(is_remote("/tmp/weights.thaw"))
        self.assertFalse(is_remote("./weights.thaw"))
        self.assertFalse(is_remote("weights.thaw"))
        self.assertFalse(is_remote(""))
        self.assertFalse(is_remote(None))

    def test_remote_uris(self):
        self.assertTrue(is_remote("s3://bucket/weights.thaw"))
        self.assertTrue(is_remote("gs://bucket/weights.thaw"))
        self.assertTrue(is_remote("https://example.com/weights.thaw"))
        self.assertTrue(is_remote("http://example.com/weights.thaw"))

    def test_unknown_scheme(self):
        # file:// and other unsupported schemes are treated as local
        self.assertFalse(is_remote("file:///tmp/weights.thaw"))
        self.assertFalse(is_remote("ftp://host/weights.thaw"))


class TestParseS3(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_parse_s3("s3://bucket/key.thaw"), ("bucket", "key.thaw"))
        self.assertEqual(
            _parse_s3("s3://my-bucket/path/to/weights.thaw"),
            ("my-bucket", "path/to/weights.thaw"),
        )

    def test_invalid(self):
        with self.assertRaises(ValueError):
            _parse_s3("s3://bucket")  # no key
        with self.assertRaises(ValueError):
            _parse_s3("s3:///key")    # no bucket


class TestCachePath(unittest.TestCase):
    def test_deterministic(self):
        uri = "s3://bucket/weights.thaw"
        p1 = _cache_path(uri, "/tmp/cache")
        p2 = _cache_path(uri, "/tmp/cache")
        self.assertEqual(p1, p2)

    def test_distinct_per_uri(self):
        p1 = _cache_path("s3://bucket/a.thaw", "/tmp/cache")
        p2 = _cache_path("s3://bucket/b.thaw", "/tmp/cache")
        self.assertNotEqual(p1, p2)

    def test_preserves_basename(self):
        path = _cache_path("s3://bucket/weights.rank1.thaw", "/tmp/cache")
        self.assertTrue(path.endswith("weights.rank1.thaw"))


class TestResolveSnapshotPath(unittest.TestCase):
    def test_local_passthrough(self):
        self.assertEqual(resolve_snapshot_path("/tmp/x.thaw"), "/tmp/x.thaw")
        self.assertEqual(resolve_snapshot_path(""), "")
        self.assertEqual(resolve_snapshot_path(None), None)

    @patch("thaw_common.cloud._s3_client")
    def test_s3_download_small_file(self, mock_client):
        """Small files (below multipart_threshold) take the single-GET path."""
        with tempfile.TemporaryDirectory() as cache_dir:
            payload = b"fake-snapshot-bytes"
            mock_client.return_value = _mock_s3_with_payload(payload)

            local = resolve_snapshot_path(
                "s3://my-bucket/weights.thaw", cache_dir=cache_dir
            )

            self.assertTrue(os.path.exists(local))
            self.assertTrue(local.endswith("weights.thaw"))
            with open(local, "rb") as f:
                self.assertEqual(f.read(), payload)

    @patch("thaw_common.cloud._s3_client")
    def test_s3_download_ranged(self, mock_client):
        """Files above multipart_threshold trigger parallel ranged GETs."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Build a payload larger than threshold (16 MiB by default).
            # Use 40 MiB of deterministic bytes so reassembly is verifiable.
            size = 40 * 1024 * 1024
            payload = bytes((i * 7 + 13) & 0xFF for i in range(size))
            mock_client.return_value = _mock_s3_with_payload(payload)

            local = resolve_snapshot_path(
                "s3://bucket/big.thaw", cache_dir=cache_dir
            )

            with open(local, "rb") as f:
                got = f.read()
            self.assertEqual(len(got), size)
            self.assertEqual(got, payload)
            # At least 2 ranged GETs should have been issued.
            calls_with_range = [
                c for c in mock_client.return_value.get_object.call_args_list
                if c.kwargs.get("Range")
            ]
            self.assertGreaterEqual(len(calls_with_range), 2)

    @patch("thaw_common.cloud._s3_client")
    def test_cache_hit_skips_download(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_client.return_value = _mock_s3_with_payload(b"x")

            p1 = resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir)
            p2 = resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir)

            self.assertEqual(p1, p2)
            # Second call should not re-download.
            self.assertEqual(mock_client.return_value.head_object.call_count, 1)

    @patch("thaw_common.cloud._s3_client")
    def test_force_bypasses_cache(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_client.return_value = _mock_s3_with_payload(b"x")

            resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir)
            resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir, force=True)

            self.assertEqual(mock_client.return_value.head_object.call_count, 2)

    def test_unsupported_scheme(self):
        with self.assertRaises(NotImplementedError):
            resolve_snapshot_path("gs://bucket/w.thaw")


class TestUploadSnapshot(unittest.TestCase):
    def test_rejects_local_destination(self):
        with self.assertRaises(ValueError):
            upload_snapshot("/tmp/x.thaw", "/tmp/y.thaw")

    @unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
    @patch("thaw_common.cloud._s3_client")
    def test_s3_upload(self, mock_client):
        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"snapshot-bytes")
            local = f.name

        try:
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3

            upload_snapshot(local, "s3://my-bucket/weights.thaw")

            # upload_file now takes a Config kwarg for multipart tuning.
            mock_s3.upload_file.assert_called_once()
            args, kwargs = mock_s3.upload_file.call_args
            self.assertEqual(args[0], local)
            self.assertEqual(args[1], "my-bucket")
            self.assertEqual(args[2], "weights.thaw")
            self.assertIn("Config", kwargs)
        finally:
            os.unlink(local)


@unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
class TestTypedErrors(unittest.TestCase):
    """Verify boto3 error codes map to the right Python exception types.

    The audit finding was that users saw raw boto3 tracebacks on common
    failures (missing key, bad creds, 403). These tests pin down the
    mapping so regressions surface as failures here, not as production
    incidents.
    """

    def _fake_client_error(self, code, status=None):
        from botocore.exceptions import ClientError
        err = {"Error": {"Code": code}}
        if status is not None:
            err["ResponseMetadata"] = {"HTTPStatusCode": status}
        return ClientError(err, "HeadObject")

    @patch("thaw_common.cloud._s3_client")
    def test_missing_key_raises_file_not_found(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.head_object.side_effect = self._fake_client_error("NoSuchKey", 404)
            mock_client.return_value = mock_s3

            with self.assertRaises(FileNotFoundError):
                resolve_snapshot_path("s3://b/missing.thaw", cache_dir=cache_dir)

    @patch("thaw_common.cloud._s3_client")
    def test_403_raises_permission_error(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.head_object.side_effect = self._fake_client_error("AccessDenied", 403)
            mock_client.return_value = mock_s3

            with self.assertRaises(PermissionError):
                resolve_snapshot_path("s3://b/blocked.thaw", cache_dir=cache_dir)

    @patch("thaw_common.cloud._s3_client")
    def test_no_credentials_raises_runtime_error(self, mock_client):
        from botocore.exceptions import NoCredentialsError

        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.head_object.side_effect = NoCredentialsError()
            mock_client.return_value = mock_s3

            with self.assertRaises(RuntimeError) as cm:
                resolve_snapshot_path("s3://b/x.thaw", cache_dir=cache_dir)
            self.assertIn("credentials", str(cm.exception).lower())

    @patch("thaw_common.cloud._s3_client")
    def test_generic_client_error_raises_runtime_error(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.head_object.side_effect = self._fake_client_error("InternalError", 500)
            mock_client.return_value = mock_s3

            with self.assertRaises(RuntimeError):
                resolve_snapshot_path("s3://b/x.thaw", cache_dir=cache_dir)


@unittest.skipUnless(HAVE_MOTO, "moto not installed")
class TestS3RoundTripMoto(unittest.TestCase):
    """End-to-end round trip against moto's in-process S3 server.

    Unlike the MagicMock tests above, this actually runs boto3's
    HTTP/XML code path — so bucket/key parsing, Range headers, and
    error mapping are validated for real.
    """

    def setUp(self):
        # Each method runs under its own @mock_aws backend. Drop any cached
        # S3 client between methods so a client built under one method's mock
        # context can't leak into the next (defense-in-depth for the shared
        # client cache; keeps these green regardless of test ordering).
        reset_s3_client_cache()

    def tearDown(self):
        reset_s3_client_cache()

    @mock_aws
    def test_round_trip_small(self):
        bucket = "thaw-test-bucket"
        key = "weights.thaw"
        payload = b"THAW\x00\x00\x00\x01" + os.urandom(4096)

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(payload)
            local_src = f.name

        with tempfile.TemporaryDirectory() as cache_dir:
            try:
                uri = f"s3://{bucket}/{key}"
                upload_snapshot(local_src, uri)

                downloaded = resolve_snapshot_path(uri, cache_dir=cache_dir)
                self.assertTrue(os.path.exists(downloaded))
                with open(downloaded, "rb") as f:
                    self.assertEqual(f.read(), payload)

                # Second resolve is a cache hit — no re-download needed.
                again = resolve_snapshot_path(uri, cache_dir=cache_dir)
                self.assertEqual(again, downloaded)
            finally:
                os.unlink(local_src)

    @mock_aws
    def test_round_trip_large_ranged(self):
        """Upload 40 MiB, download via parallel ranged GETs, verify bit-identical."""
        bucket = "thaw-big-bucket"
        key = "big.thaw"
        payload = os.urandom(40 * 1024 * 1024)

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(payload)
            local_src = f.name

        with tempfile.TemporaryDirectory() as cache_dir:
            try:
                uri = f"s3://{bucket}/{key}"
                upload_snapshot(local_src, uri)

                downloaded = resolve_snapshot_path(uri, cache_dir=cache_dir)
                with open(downloaded, "rb") as f:
                    self.assertEqual(f.read(), payload)
            finally:
                os.unlink(local_src)

    @mock_aws
    def test_missing_bucket_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_snapshot_path(
                    "s3://does-not-exist/x.thaw", cache_dir=cache_dir
                )

    @mock_aws
    def test_missing_key_raises_file_not_found(self):
        bucket = "thaw-test-bucket"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

        with tempfile.TemporaryDirectory() as cache_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_snapshot_path(
                    f"s3://{bucket}/does-not-exist.thaw", cache_dir=cache_dir
                )


class TestPoolRegisterRemote(unittest.TestCase):
    """Audit fix: pool.register() must accept s3:// URIs without hitting
    os.path.exists(), which always returns False for them."""

    def test_register_accepts_s3_uri(self):
        from thaw_vllm.pool import EnginePool

        pool = EnginePool()
        # No download happens at register time — this call should succeed
        # without any S3 credentials or network.
        pool.register("remote-model", "s3://my-bucket/weights.thaw")

        self.assertEqual(
            pool.snapshots["remote-model"], "s3://my-bucket/weights.thaw"
        )

    def test_register_still_rejects_bad_local(self):
        from thaw_vllm.pool import EnginePool

        pool = EnginePool()
        with self.assertRaises(FileNotFoundError):
            pool.register("bad", "/definitely/not/a/real/path.thaw")


class TestSharedClientCache(unittest.TestCase):
    """The S3 client is built once and reused, keyed by (pid, pool_size).

    These tests stub _build_s3_client so they need neither boto3 credentials
    nor a network — they assert the CACHING contract, not boto3 itself.
    """

    def setUp(self):
        reset_s3_client_cache()

    def tearDown(self):
        reset_s3_client_cache()

    def test_client_is_reused_across_calls(self):
        builds = {"n": 0}

        def fake_build(pool_size):
            builds["n"] += 1
            return MagicMock(name=f"client-{builds['n']}")

        with patch.object(cloud, "_build_s3_client", side_effect=fake_build):
            c1 = _s3_client()
            c2 = _s3_client()
            c3 = _s3_client()

        self.assertIs(c1, c2)
        self.assertIs(c2, c3)
        self.assertEqual(builds["n"], 1, "client should be built exactly once")

    def test_reset_forces_rebuild(self):
        with patch.object(
            cloud, "_build_s3_client", side_effect=lambda p: MagicMock()
        ) as build:
            _s3_client()
            _s3_client()
            self.assertEqual(build.call_count, 1)
            reset_s3_client_cache()
            _s3_client()
            self.assertEqual(build.call_count, 2)

    def test_distinct_pool_size_distinct_client(self):
        with patch.object(
            cloud, "_build_s3_client", side_effect=lambda p: MagicMock()
        ) as build:
            _s3_client(concurrency=32)   # pool 36
            _s3_client(concurrency=32)   # cache hit
            _s3_client(concurrency=128)  # pool 132 -> distinct key
            self.assertEqual(build.call_count, 2)

    def test_pool_size_matches_concurrency(self):
        seen = {}

        def fake_build(pool_size):
            seen["pool"] = pool_size
            return MagicMock()

        with patch.object(cloud, "_build_s3_client", side_effect=fake_build):
            _s3_client(concurrency=32)
        # Same pool math as the original per-call implementation: conc + 4, min 16.
        self.assertEqual(seen["pool"], 36)

    def test_fork_rebuilds_client(self):
        """A different pid must yield a different client (botocore isn't fork-safe)."""
        with patch.object(
            cloud, "_build_s3_client", side_effect=lambda p: MagicMock()
        ) as build:
            with patch.object(cloud.os, "getpid", return_value=1000):
                _s3_client()
                _s3_client()
            self.assertEqual(build.call_count, 1)
            # Simulate being in a forked child: pid changes -> rebuild.
            with patch.object(cloud.os, "getpid", return_value=2000):
                _s3_client()
            self.assertEqual(build.call_count, 2)


class TestResolveSnapshotsBatch(unittest.TestCase):
    """Bounded-concurrency batch download orchestration.

    Patches the single-file resolve_snapshot_path so these tests exercise the
    fan-out/ordering/error-propagation logic without boto3 or POSIX pwrite.
    """

    def test_empty_list(self):
        self.assertEqual(resolve_snapshots([]), [])

    def test_all_local_passthrough_no_threads(self):
        # Local paths must pass through untouched and must NOT call the S3 path.
        with patch.object(cloud, "resolve_snapshot_path") as rsp:
            out = resolve_snapshots(["/a.thaw", "", None, "./b.thaw"])
        self.assertEqual(out, ["/a.thaw", "", None, "./b.thaw"])
        rsp.assert_not_called()

    def test_order_preserved_under_concurrency(self):
        # Make earlier items take LONGER so completion order != input order;
        # the result list must still match input order.
        def fake_resolve(uri, cache_dir=None, force=False):
            idx = int(uri.rsplit("/", 1)[1].split(".")[0])
            time.sleep(0.05 * (5 - idx))  # item 0 sleeps longest
            return f"/cache/{idx}"

        uris = [f"s3://b/{i}.thaw" for i in range(5)]
        with patch.object(cloud, "resolve_snapshot_path", side_effect=fake_resolve):
            out = resolve_snapshots(uris, max_files=5)
        self.assertEqual(out, [f"/cache/{i}" for i in range(5)])

    def test_runs_concurrently(self):
        # 6 items each sleeping 0.2s: sequential would be >1.0s, 6-way ~0.2s.
        def slow_resolve(uri, cache_dir=None, force=False):
            time.sleep(0.2)
            return uri.replace("s3://b/", "/cache/")

        uris = [f"s3://b/{i}.thaw" for i in range(6)]
        with patch.object(cloud, "resolve_snapshot_path", side_effect=slow_resolve):
            t0 = time.monotonic()
            resolve_snapshots(uris, max_files=6)
            elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.8, f"expected concurrent (~0.2s), took {elapsed:.2f}s")

    def test_respects_max_files_bound(self):
        # With max_files=2, 4 items of 0.2s each => 2 waves => ~0.4s, not ~0.2s.
        def slow_resolve(uri, cache_dir=None, force=False):
            time.sleep(0.2)
            return uri
        uris = [f"s3://b/{i}.thaw" for i in range(4)]
        with patch.object(cloud, "resolve_snapshot_path", side_effect=slow_resolve):
            t0 = time.monotonic()
            resolve_snapshots(uris, max_files=2)
            elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.35, f"max_files=2 should serialize into waves, took {elapsed:.2f}s")

    def test_mixed_local_and_remote(self):
        def fake_resolve(uri, cache_dir=None, force=False):
            return "/cache/" + uri.rsplit("/", 1)[1]
        items = ["/local/a.thaw", "s3://b/x.thaw", "/local/c.thaw", "s3://b/y.thaw"]
        with patch.object(cloud, "resolve_snapshot_path", side_effect=fake_resolve):
            out = resolve_snapshots(items)
        self.assertEqual(
            out, ["/local/a.thaw", "/cache/x.thaw", "/local/c.thaw", "/cache/y.thaw"]
        )

    def test_first_error_propagates(self):
        def fake_resolve(uri, cache_dir=None, force=False):
            if uri.endswith("2.thaw"):
                raise FileNotFoundError("thaw S3 download failed: object not found at " + uri)
            time.sleep(0.05)
            return "/cache/ok"
        uris = [f"s3://b/{i}.thaw" for i in range(4)]
        with patch.object(cloud, "resolve_snapshot_path", side_effect=fake_resolve):
            with self.assertRaises(FileNotFoundError):
                resolve_snapshots(uris, max_files=4)

    def test_passes_cache_dir_and_force(self):
        seen = []
        def fake_resolve(uri, cache_dir=None, force=False):
            seen.append((cache_dir, force))
            return "/cache/x"
        with patch.object(cloud, "resolve_snapshot_path", side_effect=fake_resolve):
            resolve_snapshots(["s3://b/x.thaw"], cache_dir="/custom", force=True)
        self.assertEqual(seen, [("/custom", True)])

    def test_duplicate_uris_downloaded_once(self):
        # Duplicate URIs must download exactly once (they share one cache file;
        # racing two .part writes/renames would corrupt it) and every slot must
        # receive the same resolved path, in order.
        calls = []
        def fake_resolve(uri, cache_dir=None, force=False):
            calls.append(uri)
            return "/cache/" + uri.rsplit("/", 1)[1]
        uris = [
            "s3://b/dup.thaw",
            "s3://b/other.thaw",
            "s3://b/dup.thaw",
            "s3://b/dup.thaw",
        ]
        with patch.object(cloud, "resolve_snapshot_path", side_effect=fake_resolve):
            out = resolve_snapshots(uris, max_files=8)
        self.assertEqual(out, [
            "/cache/dup.thaw", "/cache/other.thaw",
            "/cache/dup.thaw", "/cache/dup.thaw",
        ])
        self.assertEqual(sorted(calls), ["s3://b/dup.thaw", "s3://b/other.thaw"])
        self.assertEqual(calls.count("s3://b/dup.thaw"), 1)


class TestUploadSnapshotsBatch(unittest.TestCase):
    def test_empty_noop(self):
        # Must not raise and must not touch the S3 path.
        with patch.object(cloud, "upload_snapshot") as up:
            upload_snapshots([])
        up.assert_not_called()

    def test_all_uploaded(self):
        calls = []
        with patch.object(cloud, "upload_snapshot", side_effect=lambda lp, uri: calls.append((lp, uri))):
            upload_snapshots([("/a", "s3://b/a"), ("/c", "s3://b/c")])
        self.assertEqual(sorted(calls), [("/a", "s3://b/a"), ("/c", "s3://b/c")])

    def test_runs_concurrently(self):
        def slow_upload(lp, uri):
            time.sleep(0.2)
        pairs = [(f"/f{i}", f"s3://b/{i}") for i in range(6)]
        with patch.object(cloud, "upload_snapshot", side_effect=slow_upload):
            t0 = time.monotonic()
            upload_snapshots(pairs, max_files=6)
            elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.8, f"expected concurrent (~0.2s), took {elapsed:.2f}s")

    def test_first_error_propagates(self):
        def maybe_fail(lp, uri):
            if uri.endswith("bad"):
                raise PermissionError("access denied")
        pairs = [("/a", "s3://b/ok"), ("/b", "s3://b/bad"), ("/c", "s3://b/ok2")]
        with patch.object(cloud, "upload_snapshot", side_effect=maybe_fail):
            with self.assertRaises(PermissionError):
                upload_snapshots(pairs, max_files=3)


@unittest.skipUnless(HAVE_MOTO, "moto not installed")
@unittest.skipUnless(hasattr(os, "pwrite"), "ranged download path requires POSIX os.pwrite")
class TestBatchRoundTripMoto(unittest.TestCase):
    """End-to-end batch round trip against moto. POSIX-only: the download
    write path uses os.pwrite. Runs in CI (Ubuntu); skipped on Windows."""

    @mock_aws
    def test_resolve_and_upload_many(self):
        reset_s3_client_cache()
        bucket = "thaw-batch-bucket"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

        payloads = {f"obj{i}.thaw": os.urandom(2048 + i) for i in range(6)}
        srcs = []
        try:
            pairs = []
            for name, data in payloads.items():
                fd, p = tempfile.mkstemp(suffix=".thaw")
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                srcs.append(p)
                pairs.append((p, f"s3://{bucket}/{name}"))

            # Batch upload, then batch download, verify bit-identical + order.
            upload_snapshots(pairs)
            with tempfile.TemporaryDirectory() as cache_dir:
                uris = [f"s3://{bucket}/{n}" for n in payloads]
                locals_ = resolve_snapshots(uris, cache_dir=cache_dir)
                self.assertEqual(len(locals_), len(uris))
                for (name, data), local in zip(payloads.items(), locals_):
                    with open(local, "rb") as f:
                        self.assertEqual(f.read(), data, f"mismatch for {name}")
        finally:
            for p in srcs:
                if os.path.exists(p):
                    os.unlink(p)

    @mock_aws
    def test_batch_surfaces_missing_object(self):
        reset_s3_client_cache()
        bucket = "thaw-batch-bucket2"
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        with tempfile.TemporaryDirectory() as cache_dir:
            uris = [f"s3://{bucket}/missing-{i}.thaw" for i in range(3)]
            with self.assertRaises(FileNotFoundError):
                resolve_snapshots(uris, cache_dir=cache_dir)

    @mock_aws
    def test_duplicate_uris_no_race(self):
        # Real-path regression for the duplicate-URI race: without dedup, two
        # threads resolving the same URI race the same .part rename (torn cache
        # on POSIX, FileExistsError on Windows). With dedup it downloads once
        # and every slot gets the same byte-identical file.
        reset_s3_client_cache()
        bucket = "thaw-dup-bucket"
        key = "shared.thaw"
        payload = os.urandom(8192)
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=payload)

        uri = f"s3://{bucket}/{key}"
        with tempfile.TemporaryDirectory() as cache_dir:
            out = resolve_snapshots([uri] * 5, cache_dir=cache_dir, max_files=8)
            self.assertEqual(len(out), 5)
            self.assertEqual(len(set(out)), 1)  # all slots -> same cache file
            with open(out[0], "rb") as f:
                self.assertEqual(f.read(), payload)


if __name__ == "__main__":
    unittest.main()
