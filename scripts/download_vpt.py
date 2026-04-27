"""Download VPT model definitions and pretrained weights.

OpenAI hosts the VPT artifacts on a public CDN. We download the matching
`.model` (architecture description) and `.weights` (parameters) into
`weights/vpt/`. URLs follow the pattern documented in:
https://github.com/openai/Video-Pre-Training#agent-model-files

Usage:
    python scripts/download_vpt.py --size 2x
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://openaipublic.blob.core.windows.net/minecraft-rl/models"

ARTIFACTS = {
    "1x": {
        "model": f"{BASE}/foundation-model-1x.model",
        "weights": f"{BASE}/foundation-model-1x.weights",
    },
    "2x": {
        "model": f"{BASE}/foundation-model-2x.model",
        "weights": f"{BASE}/foundation-model-2x.weights",
    },
    "3x": {
        "model": f"{BASE}/foundation-model-3x.model",
        "weights": f"{BASE}/foundation-model-3x.weights",
    },
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100 * downloaded / total
                print(f"  {downloaded / 1e6:8.1f} / {total / 1e6:8.1f} MB ({pct:5.1f}%)", end="\r")
        print()
    tmp.rename(dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"[sha ] {dest.name}  sha256={h[:16]}...")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=sorted(ARTIFACTS), default="2x")
    ap.add_argument("--out", default="weights/vpt", help="Output directory.")
    args = ap.parse_args()

    out = Path(args.out)
    artifacts = ARTIFACTS[args.size]
    for kind, url in artifacts.items():
        ext = ".model" if kind == "model" else ".weights"
        _download(url, out / f"foundation-model-{args.size}{ext}")
    print(f"\nVPT-{args.size} ready in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
