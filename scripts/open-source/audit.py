"""Local pre-publication evidence. Does not approve or publish a release.

Python 3.11+, Rust, Git. Optional scanners can be found on PATH or --tools.
Outputs are internal; commands never upload repository source to a service.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE = ("docs/open-source/", "target/", ".git/", "private/")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def snapshot(destination, exclude=EXCLUDE):
    names = sorted(set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8").split("\0")) - {""})
    records = []
    for name in names:
        if name.startswith(exclude):
            continue
        source = ROOT / name
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Review link before exporting: {name}")
        data = source.read_bytes()  # Deleted tracked files fail closed.
        dest = destination / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        records.append({"path": name, "sha256": digest(data), "bytes": len(data)})
    return records


def find_tool(name, toolroot):
    installed = shutil.which(name)
    if installed:
        return installed
    if toolroot:
        matches = list(toolroot.rglob(name + (".exe" if os.name == "nt" else "")))
        if matches:
            return str(sorted(matches)[0])
    return None


def run_check(reports, name, argv, cwd, env=None, timeout=1200):
    start = time.monotonic()
    result = {"name": name, "command": argv, "cwd": str(cwd)}
    if not argv[0]:
        result.update(status="unavailable", exit_code=None)
    else:
        try:
            with (reports / (name + ".stdout.log")).open("wb") as out, (reports / (name + ".stderr.log")).open("wb") as err:
                p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err)
                try:
                    code = p.wait(timeout=timeout)
                    result.update(status="passed" if code == 0 else "failed", exit_code=code)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                    else:
                        p.kill()
                    p.wait()
                    result.update(status="timeout", exit_code=None)
        except OSError as exc:
            result.update(status="unavailable", exit_code=None, error=str(exc))
    result["seconds"] = round(time.monotonic() - start, 2)
    write_json(reports / (name + ".result.json"), result)
    print(name, result["status"], flush=True)
    return result


def inventory(run, records):
    reports = run / "reports"
    files = []
    for r in records:
        path = run / "source" / r["path"]
        data = path.read_bytes()
        if path.suffix in {".lib", ".a", ".whl"}:
            strings = re.findall(rb"[\x20-\x7e]{12,}", data)
            sensitive = [s.decode("ascii")[:500] for s in strings if re.search(rb"(?:[A-Za-z]:[\\/](?:Users|repositories)|/home/|/Users/|/root/|private[/\\]arcqml)", s)]
            files.append({**r, "embedded_path_samples": sorted(set(sensitive))[:30], "runtime_dependency_inventory": "See accompanying BUILD-METADATA.json and runtime.cdx.json; file presence alone does not verify correspondence."})
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as z:
                items = []
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    dest = run / "wheel-contents" / path.name / member.filename
                    if not dest.resolve().is_relative_to((run / "wheel-contents").resolve()):
                        raise ValueError("Unsafe wheel member")
                    b = z.read(member)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b)
                    items.append({"path": member.filename, "bytes": len(b), "sha256": digest(b)})
                write_json(reports / (path.name + ".contents.json"), items)
    write_json(reports / "binary-inventory.json", files)
    provenance = [r | {"origin_status": "needs-review"} for r in records if r["path"].startswith("examples/data/") or r["path"].endswith((".svg", ".png", ".woff2", ".ttf"))]
    write_json(reports / "data-assets.json", provenance)
    patterns = {
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "internal-path": r"[A-Za-z]:[\\/](?:Users|repositories)[^\s\"<>]{1,160}|/(?:home|Users|root)/[^\s\"<>]{1,160}",
        "private-ip": r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        "origin-reference": r"https?://[^\s\"<>]{1,200}|(?i:copyright|licensed under|derived from|adapted from)",
    }
    hits = []
    for r in records:
        data = (run / "source" / r["path"]).read_bytes()
        if b"\0" in data or len(data) > 5000000:
            continue
        text = data.decode("utf-8", errors="replace")
        for kind, pattern in patterns.items():
            for m in re.finditer(pattern, text):
                hits.append({"path": r["path"], "kind": kind, "line": text.count("\n", 0, m.start()) + 1, "match": m.group(0)})
    write_json(reports / "content-review.json", hits)
    # Inspect extracted wheel binaries as well as the SDK archives. Scanning
    # printable strings is supplementary evidence, not binary decompilation.
    binary_paths = list((run / "source/libs").rglob("*.lib")) + list((run / "source/libs").rglob("*.a"))
    binary_paths += list((run / "wheel-contents").rglob("*.pyd")) + list((run / "wheel-contents").rglob("*.so"))
    string_dir = run / "binary-strings"
    string_dir.mkdir(exist_ok=True)
    binary_records = []
    for i, path in enumerate(binary_paths):
        data = path.read_bytes()
        strings = re.findall(rb"[\x20-\x7e]{8,}", data)
        wide = [s.decode("utf-16-le").encode() for s in re.findall(rb"(?:[\x20-\x7e]\x00){8,}", data)]
        (string_dir / f"{i}.txt").write_bytes(b"\n".join(strings + wide))
        text = b"\n".join(strings)
        crates = sorted(set((a.decode(), v.decode()) for a, v in re.findall(rb"[/\\]([a-zA-Z0-9_-]+)-(\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?)[/\\]src[/\\]", text)))
        paths = [s.decode() for s in strings if re.search(rb"[A-Za-z]:[\\/](?:Users|repositories)|/(?:home|Users|root)/", s)]
        binary_records.append({"path": path.relative_to(run).as_posix(), "sha256": digest(data), "string_file": f"{i}.txt", "observed_crate_versions": crates, "internal_path_count": len(paths), "internal_path_samples": paths[:15], "coverage": "Observed strings only; not a complete binary dependency inventory"})
    write_json(reports / "binary-string-evidence.json", binary_records)


def sbom(run):
    reports = run / "reports"
    meta = json.loads((reports / "cargo-metadata.stdout.log").read_text(encoding="utf-8"))
    lock = tomllib.loads((run / "source/Cargo.lock").read_text(encoding="utf-8"))
    by_key = {(p["name"], p["version"], p.get("source")): p for p in meta["packages"]}
    components, refs, license_rows = [], {}, []
    for p in lock["package"]:
        key = (p["name"], p["version"], p.get("source"))
        m = by_key.get(key)
        ref = f"pkg:cargo/{p['name']}@{p['version']}"
        if p.get("source") and not p["source"].startswith("registry+"):
            ref += "?source=" + digest(p["source"].encode())[:16]
        refs[key] = ref
        c = {"type": "library", "bom-ref": ref, "name": p["name"], "version": p["version"], "purl": ref}
        if p.get("checksum"):
            c["hashes"] = [{"alg": "SHA-256", "content": p["checksum"]}]
        if m and m.get("license"):
            c["licenses"] = [{"expression": m["license"].replace("/", " OR ")}]
        c["properties"] = [{"name": "arcqml:inventory-scope", "value": "workspace-source" if not p.get("source") else "lockfile-dependency"}]
        components.append(c)
        texts = []
        if m and p.get("source"):
            base = Path(m["manifest_path"]).parent
            candidates = [f for f in base.rglob("*") if f.is_file() and re.match(r"(?i)^(LICENSE|COPYING|NOTICE|COPYRIGHT|AUTHORS)(?:[._-].*)?$", f.name)]
            for f in candidates:
                rel = Path(p["name"] + "-" + p["version"]) / f.relative_to(base)
                dst = reports / "third-party-license-texts" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
                texts.append({"path": rel.as_posix(), "sha256": digest(f.read_bytes())})
        license_rows.append({"name": p["name"], "version": p["version"], "source": p.get("source"), "license": m.get("license") if m else None, "metadata_present": m is not None, "license_files": texts})
    ids = {m["id"]: refs[(m["name"], m["version"], m.get("source"))] for m in meta["packages"]}
    edges = [{"ref": ids[n["id"]], "dependsOn": sorted({ids[d["pkg"]] for d in n["deps"]})} for n in meta["resolve"]["nodes"]]
    seen = {e["ref"] for e in edges}
    edges += [{"ref": c["bom-ref"], "dependsOn": []} for c in components if c["bom-ref"] not in seen]
    bom = {"bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": "urn:uuid:" + str(uuid.uuid4()), "version": 1,
           "metadata": {"timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "component": {"type": "application", "name": "ArcQML-source-review", "version": "0.1.0", "bom-ref": "arcqml-source"},
                        "properties": [{"name": "arcqml:source-manifest-sha256", "value": digest((reports / "source-files.json").read_bytes())},
                                       {"name": "arcqml:coverage", "value": "Cargo.lock, all features, all target dependencies including dev/build. Excludes hidden Runtime build dependencies and Python environment; not a complete binary SBOM."}]},
           "components": components, "dependencies": [{"ref": "arcqml-source", "dependsOn": [ids[i] for i in meta["workspace_members"]]}] + edges}
    write_json(reports / "workspace-source.cdx.json", bom)
    write_json(reports / "dependency-licenses.json", license_rows)
    write_json(reports / "sbom-coverage.json", {"lock_packages": len(lock["package"]), "metadata_packages": len(meta["packages"]), "components": len(components), "workspace_packages": len(meta["workspace_members"]), "missing_metadata": [r for r in license_rows if not r["metadata_present"]]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", type=Path, help="Directory with optional gitleaks/cargo-deny/cargo-audit executables")
    ap.add_argument("--python", type=Path, help="Python for linking PyO3, preferably CPython 3.11")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--toolchain", help="Rust toolchain matching the shipped Runtime; e.g. 1.98.0")
    ap.add_argument("--windows-preview", action="store_true", help="Export Windows preview; omit unverified Linux binaries and generated rustdoc")
    ap.add_argument("--scancode", action="store_true", help="Also run ScanCode on source and extracted wheels")
    args = ap.parse_args()
    run = ROOT / "target/open-source-audit/runs" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports = run / "reports"
    reports.mkdir(parents=True)
    (ROOT / "target/open-source-audit/latest.txt").write_text(str(run), encoding="utf-8")
    exclude = EXCLUDE + (("libs/x86_64-unknown-linux-gnu/", "wheels/x86_64-unknown-linux-gnu/", "docs/doc/") if args.windows_preview else ())
    records = snapshot(run / "source", exclude)
    write_json(reports / "source-files.json", records)
    write_json(reports / "context.json", {"head": git("rev-parse", "HEAD").decode().strip(), "git_refs": git("for-each-ref", "--format=%(refname) %(objectname)").decode(), "shallow": git("rev-parse", "--is-shallow-repository").decode().strip(), "commits": int(git("rev-list", "--all", "--count")), "status": git("status", "--porcelain=v1").decode("utf-8"), "source_files": len(records), "excluded": exclude, "approval": "NOT_GRANTED_BY_THIS_TOOL"})
    (reports / "authors-internal.txt").write_bytes(git("log", "--all", "--format=%an <%ae>%n%cn <%ce>"))
    inventory(run, records)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(run / "build")
    env["ARCQML_RUNTIME_LIB_DIR"] = str(run / "source/libs/x86_64-pc-windows-msvc")
    if args.python:
        env["PYO3_PYTHON"] = str(args.python.resolve())
    if args.toolchain:
        env["RUSTUP_TOOLCHAIN"] = args.toolchain
    deny = find_tool("cargo-deny", args.tools)
    leaks = find_tool("gitleaks", args.tools)
    audit = find_tool("cargo-audit", args.tools)
    jobs = [
        ("cargo-metadata", ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1"], run / "source"),
        ("gitleaks-history", [leaks, "git", str(ROOT), "--log-opts=--all --full-history", "--redact=100", "--no-banner", "--ignore-gitleaks-allow", "--report-format=json", "--report-path=" + str(reports / "gitleaks-history.json")], ROOT),
        ("gitleaks-current", [leaks, "dir", str(run / "source"), "--redact=100", "--no-banner", "--ignore-gitleaks-allow", "--report-format=json", "--report-path=" + str(reports / "gitleaks-current.json")], ROOT),
        ("gitleaks-wheels", [leaks, "dir", str(run / "wheel-contents"), "--redact=100", "--no-banner", "--ignore-gitleaks-allow", "--report-format=json", "--report-path=" + str(reports / "gitleaks-wheels.json")], ROOT),
        ("cargo-audit", [audit, "audit", "--file", str(run / "source/Cargo.lock"), "--json"], run / "source"),
        ("cargo-deny", [deny, "--all-features", "--locked", "--format", "json", "--config", "scripts/open-source/deny.toml", "check", "licenses", "sources", "bans"], run / "source"),
        ("cargo-deny-list", [deny, "--all-features", "--locked", "--config", "scripts/open-source/deny.toml", "list", "--format", "json", "--layout", "crate"], run / "source"),
        ("gitleaks-binary-strings", [leaks, "dir", str(run / "binary-strings"), "--redact=100", "--no-banner", "--ignore-gitleaks-allow", "--report-format=json", "--report-path=" + str(reports / "gitleaks-binary-strings.json")], ROOT),
    ]
    if args.scancode:
        scanner = find_tool("scancode", args.tools)
        supplement = run / "supplement"
        supplement.mkdir()
        shutil.copyfile(run / "source/.gitignore", supplement / "gitignore.txt")
        for name, directory in [("scancode-source", run / "source"), ("scancode-supplement", supplement), ("scancode-wheels", run / "wheel-contents")]:
            jobs.append((name, [scanner, "--license", "--copyright", "--package", "--info", "--license-text", "--processes", "2", "--json-pp", str(reports / (name + ".json")), str(directory)], ROOT))
    checks = []
    for name, argv in [("cargo-version", ["cargo", "-V"]), ("rustc-version", ["rustc", "-Vv"]),
                       ("gitleaks-version", [leaks, "version"]), ("cargo-deny-version", [deny, "--version"]),
                       ("cargo-audit-version", [audit, "audit", "--version"])]:
        checks.append(run_check(reports, name, argv, run / "source", env))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_check, reports, name, argv, cwd, env) for name, argv, cwd in jobs]
        checks.extend(f.result() for f in futures)
    if next(c for c in checks if c["name"] == "cargo-metadata")["exit_code"] == 0:
        sbom(run)
    if not args.scan_only:
        commands = [
            ("cargo-fmt", ["cargo", "fmt", "--all", "--", "--check"]),
            ("cargo-check", ["cargo", "check", "--workspace", "--all-targets", "--all-features", "--locked"]),
            ("cargo-test", ["cargo", "test", "--workspace", "--locked", "--no-fail-fast"]),
            ("cargo-clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"]),
            ("cargo-doc", ["cargo", "doc", "--workspace", "--all-features", "--locked", "--no-deps"]),
        ]
        for name, argv in commands:
            checks.append(run_check(reports, name, argv, run / "source", env))
    write_json(reports / "summary.json", {"checks": checks, "release_ready": False, "reason": "Scan results require source/binary licensing, provenance, coverage and ownership review; see internal report."})
    print("Evidence:", run, flush=True)
    return 1 if any(c["status"] != "passed" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
