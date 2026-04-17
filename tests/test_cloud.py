"""
Tests for thaw_common.cloud — S3 URI resolution and cache behavior.

Two layers of coverage:
  - MagicMock-patched tests exercise the dispatch logic without boto3
  - moto `@mock_aws` tests exercise the actual boto3 code path against
    an in-process fake S3 server, so we validate the bucket/key parsing
    and error mapping instead of trusting a mock's call_args
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import boto3
    from moto import mock_aws
    HAVE_MOTO = True
except ImportError:
    HAVE_MOTO = False

# Ensure python/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from thaw_common.cloud import (
    is_remote,
    resolve_snapshot_path,
    upload_snapshot,
    _cache_path,
    _parse_s3,
)


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
    def test_s3_download(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()

            def fake_download(bucket, key, dest):
                with open(dest, "wb") as f:
                    f.write(b"fake-snapshot-bytes")

            mock_s3.download_file.side_effect = fake_download
            mock_client.return_value = mock_s3

            local = resolve_snapshot_path(
                "s3://my-bucket/weights.thaw", cache_dir=cache_dir
            )

            self.assertTrue(os.path.exists(local))
            self.assertTrue(local.endswith("weights.thaw"))
            mock_s3.download_file.assert_called_once_with(
                "my-bucket", "weights.thaw", local + ".part"
            )

    @patch("thaw_common.cloud._s3_client")
    def test_cache_hit_skips_download(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()

            def fake_download(bucket, key, dest):
                with open(dest, "wb") as f:
                    f.write(b"x")

            mock_s3.download_file.side_effect = fake_download
            mock_client.return_value = mock_s3

            p1 = resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir)
            p2 = resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir)

            self.assertEqual(p1, p2)
            # Second call should not download again
            self.assertEqual(mock_s3.download_file.call_count, 1)

    @patch("thaw_common.cloud._s3_client")
    def test_force_bypasses_cache(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()

            def fake_download(bucket, key, dest):
                with open(dest, "wb") as f:
                    f.write(b"x")

            mock_s3.download_file.side_effect = fake_download
            mock_client.return_value = mock_s3

            resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir)
            resolve_snapshot_path("s3://b/w.thaw", cache_dir=cache_dir, force=True)

            self.assertEqual(mock_s3.download_file.call_count, 2)

    def test_unsupported_scheme(self):
        with self.assertRaises(NotImplementedError):
            resolve_snapshot_path("gs://bucket/w.thaw")


class TestUploadSnapshot(unittest.TestCase):
    def test_rejects_local_destination(self):
        with self.assertRaises(ValueError):
            upload_snapshot("/tmp/x.thaw", "/tmp/y.thaw")

    @patch("thaw_common.cloud._s3_client")
    def test_s3_upload(self, mock_client):
        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"snapshot-bytes")
            local = f.name

        try:
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3

            upload_snapshot(local, "s3://my-bucket/weights.thaw")

            mock_s3.upload_file.assert_called_once_with(
                local, "my-bucket", "weights.thaw"
            )
        finally:
            os.unlink(local)


class TestTypedErrors(unittest.TestCase):
    """Verify boto3 error codes map to the right Python exception types.

    The audit finding was that users saw raw boto3 tracebacks on common
    failures (missing key, bad creds, 403). These tests pin down the
    mapping so regressions surface as failures here, not as production
    incidents.
    """

    def _fake_client_error(self, code):
        from botocore.exceptions import ClientError
        return ClientError({"Error": {"Code": code}}, "GetObject")

    @patch("thaw_common.cloud._s3_client")
    def test_missing_key_raises_file_not_found(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.download_file.side_effect = self._fake_client_error("NoSuchKey")
            mock_client.return_value = mock_s3

            with self.assertRaises(FileNotFoundError):
                resolve_snapshot_path("s3://b/missing.thaw", cache_dir=cache_dir)

    @patch("thaw_common.cloud._s3_client")
    def test_403_raises_permission_error(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.download_file.side_effect = self._fake_client_error("AccessDenied")
            mock_client.return_value = mock_s3

            with self.assertRaises(PermissionError):
                resolve_snapshot_path("s3://b/blocked.thaw", cache_dir=cache_dir)

    @patch("thaw_common.cloud._s3_client")
    def test_no_credentials_raises_runtime_error(self, mock_client):
        from botocore.exceptions import NoCredentialsError

        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.download_file.side_effect = NoCredentialsError()
            mock_client.return_value = mock_s3

            with self.assertRaises(RuntimeError) as cm:
                resolve_snapshot_path("s3://b/x.thaw", cache_dir=cache_dir)
            self.assertIn("credentials", str(cm.exception).lower())

    @patch("thaw_common.cloud._s3_client")
    def test_generic_client_error_raises_runtime_error(self, mock_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            mock_s3 = MagicMock()
            mock_s3.download_file.side_effect = self._fake_client_error("InternalError")
            mock_client.return_value = mock_s3

            with self.assertRaises(RuntimeError):
                resolve_snapshot_path("s3://b/x.thaw", cache_dir=cache_dir)


@unittest.skipUnless(HAVE_MOTO, "moto not installed")
class TestS3RoundTripMoto(unittest.TestCase):
    """End-to-end round trip against moto's in-process S3 server.

    Unlike the MagicMock tests above, this actually runs boto3's
    HTTP/XML code path — so bucket/key parsing, URL construction, and
    the download/upload signatures are validated for real.
    """

    @mock_aws
    def test_round_trip(self):
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


if __name__ == "__main__":
    unittest.main()
