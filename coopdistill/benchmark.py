"""Load and verify the fixed default evaluation benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .environment import Config
from .hard_cases import case_from_dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_default_benchmark(root: Path):
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Benchmark protocol/manifest mismatch")
    cases = [case_from_dict(item["case"]) for item in manifest["cases"]]
    expected = [item["pg"]["fingerprint"] for item in manifest["cases"]]
    if [case.fingerprint for case in cases] != expected:
        raise ValueError("Benchmark physical definitions changed")
    return Config(**protocol["environment"]), manifest, cases
