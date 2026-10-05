from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import fetch_model


class Sha256FileTests(unittest.TestCase):
    def test_empty_file_matches_known_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.bin"
            empty.write_bytes(b"")
            self.assertEqual(
                fetch_model.sha256_file(empty),
                "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
            )


class CatalogTests(unittest.TestCase):
    def test_all_entries_pin_a_sha256_and_size(self) -> None:
        for name, spec in fetch_model.MODELS.items():
            with self.subTest(variant=name):
                self.assertRegex(str(spec["sha256"]), r"^[0-9A-F]{64}$")
                self.assertGreater(int(spec["size"]), 0)

    def test_describe_variants_lists_every_variant(self) -> None:
        description = fetch_model.describe_variants()
        for name in fetch_model.MODELS:
            self.assertIn(name, description)


class FetchModelTests(unittest.TestCase):
    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fetch_model.fetch_model("nope", Path(tmp) / "model.task")

    def test_existing_valid_file_is_kept_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.task"
            dest.write_bytes(b"already-downloaded")
            spec = {
                "file": "x.task",
                "url": "http://invalid.invalid/x.task",
                "sha256": fetch_model.sha256_file(dest),
                "size": dest.stat().st_size,
            }
            original = dict(fetch_model.MODELS)
            fetch_model.MODELS["test"] = spec
            try:
                result = fetch_model.fetch_model("test", dest)
            finally:
                fetch_model.MODELS.clear()
                fetch_model.MODELS.update(original)
            self.assertEqual(result, dest)


if __name__ == "__main__":
    unittest.main()
