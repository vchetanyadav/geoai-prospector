"""Fetch public pre-competitive data and detect whether anything changed upstream.

The manifest (data/raw/real/manifest.json) stores ETag / Last-Modified / sha256 per
source. The scheduled workflow calls `check_for_updates`; the expensive pipeline only
re-runs when a source actually changed. That is the "automation saves manual hours" story.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(cfg: dict) -> Path:
    return Path(cfg["_root"]) / cfg["paths"]["raw"] / "real" / "manifest.json"


def load_manifest(cfg: dict) -> dict:
    p = _manifest_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def check_for_updates(cfg: dict, timeout=30) -> dict[str, bool]:
    """HEAD every source that has a url; True means it is new or changed."""
    manifest = load_manifest(cfg)
    changed = {}
    for name, src in cfg["sources"]["real"].items():
        url = src.get("url")
        if not url:
            changed[name] = False
            continue
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            sig = r.headers.get("ETag") or r.headers.get("Last-Modified") or r.headers.get("Content-Length")
        except requests.RequestException as e:
            log.warning("could not reach %s (%s)", url, e)
            changed[name] = False
            continue
        changed[name] = manifest.get(name, {}).get("signature") != sig
    return changed


def download_all(cfg: dict, force=False, timeout=120) -> dict:
    root = Path(cfg["_root"])
    manifest = load_manifest(cfg)
    for name, src in cfg["sources"]["real"].items():
        url, dest = src.get("url"), root / src["path"]
        if not url:
            if not dest.exists():
                log.warning("%s: no url configured and %s missing - download it manually (see docs/DATA.md)",
                            name, dest)
            continue
        if dest.exists() and not force:
            log.info("%s: already present", name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("%s: downloading %s", name, url)
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            sig = r.headers.get("ETag") or r.headers.get("Last-Modified") or r.headers.get("Content-Length")
        manifest[name] = {"url": url, "signature": sig, "sha256": _sha256(dest)}
    p = _manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    return manifest
