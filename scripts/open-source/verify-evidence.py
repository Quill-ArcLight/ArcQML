"""Verify a review snapshot and create a local NOT-FOR-RELEASE archive.

This deliberately never marks legal/source/binary review as approved.
The archive contains the audited snapshot, not whatever happens to be in HEAD.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_snapshot(source, records):
    expected = {}
    for item in records:
        name = item["path"]
        if "\\" in name or ":" in name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError("Noncanonical manifest path: " + name)
        path = source / name
        if name in expected or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Duplicate or unsafe manifest path: " + name)
        if name.startswith((".git/", "target/", "docs/open-source/", "private/")) or path.is_symlink():
            raise ValueError("Excluded path in review: " + name)
        if sha(path.read_bytes()) != item["sha256"]:
            raise ValueError("Snapshot hash mismatch: " + name)
        expected[name] = item
    # Build/test outputs may exist in the isolated checkout. Only the manifest is exported.
    return expected


def scancode_coverage(reports, records):
    scanned = {}
    errors = []
    for filename, prefix in [("scancode-source.json", "source/"), ("scancode-supplement.json", "supplement/")]:
        path = reports / filename
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for header in doc.get("headers", []):
            errors.extend(header.get("errors", []))
        for item in doc["files"]:
            if item["type"] != "file":
                continue
            name = item["path"].removeprefix(prefix)
            if filename == "scancode-supplement.json" and name == "gitignore.txt":
                name = ".gitignore"
            scanned[name] = item
            if item.get("scan_errors"):
                errors.append({"path": name, "errors": item["scan_errors"]})
    missing, mismatched = [], []
    for record in records:
        match = scanned.get(record["path"])
        if match is None:
            missing.append(record["path"])
        elif match.get("sha256") != record["sha256"] and not (
            record["bytes"] == 0 and match.get("size") == 0 and record["sha256"] == sha(b"")
        ):
            mismatched.append(record["path"])
    return {"expected": len(records), "covered": len(records) - len(missing), "missing": missing, "hash_mismatches": mismatched, "scan_errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    reports, source = run / "reports", run / "source"
    records = json.loads((reports / "source-files.json").read_text(encoding="utf-8"))
    expected = verify_snapshot(source, records)
    coverage = scancode_coverage(reports, records)
    result = {"source_manifest_sha256": sha((reports / "source-files.json").read_bytes()),
              "snapshot_hashes_verified": len(expected), "scancode": coverage, "release_ready": False,
              "release_status": "NOT_FOR_RELEASE: ownership, mixed-license terms, Runtime build provenance and data review remain external requirements"}
    if coverage["missing"] or coverage["hash_mismatches"] or coverage["scan_errors"]:
        result["archive"] = None
        code = 1
    else:
        archive = run / "ArcQML-review-NOT-FOR-RELEASE.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(expected):
                z.write(source / name, name)
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(expected)
            for name in z.namelist():
                if sha(z.read(name)) != expected[name]["sha256"]:
                    raise ValueError("Archive hash mismatch: " + name)
        result["archive"] = {"path": archive.name, "sha256": sha(archive.read_bytes()), "files": len(expected)}
        code = 0
    (reports / "snapshot-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
