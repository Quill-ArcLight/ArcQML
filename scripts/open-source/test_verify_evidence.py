import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("verify", Path(__file__).with_name("verify-evidence.py"))
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ExportProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, name, data=b"reviewed"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def test_tampering_is_rejected(self):
        record = self.record("src/lib.rs")
        (self.root / record["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verify.verify_snapshot(self.root, [record])

    def test_internal_and_private_files_are_rejected(self):
        for name in ["private/runtime.rs", "docs/open-source/approval.md", ".git/config", "target/report.json"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify.verify_snapshot(self.root, [self.record(name)])

    def test_noncanonical_paths_are_rejected(self):
        for name in ["../secret", "docs\\open-source\\approval.md", "/absolute", "C:/secret"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify.verify_snapshot(self.root, [{"path": name}])

    def test_duplicate_paths_are_rejected(self):
        item = self.record("Cargo.toml")
        with self.assertRaises(ValueError):
            verify.verify_snapshot(self.root, [item, item])

    def test_only_manifest_is_exportable(self):
        item = self.record("Cargo.toml")
        self.record("generated-test-output.txt")
        self.assertEqual(list(verify.verify_snapshot(self.root, [item])), ["Cargo.toml"])


if __name__ == "__main__":
    unittest.main()
