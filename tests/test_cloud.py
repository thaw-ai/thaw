"""
Tests for thaw_common.cloud — S3 URI resolution and cache behavior.

boto3 is mocked so these run on Mac without AWS credentials.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
