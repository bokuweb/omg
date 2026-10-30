#!/usr/bin/env python3
"""Download the official smallest Bonsai 2 4B packing, without repacking weights."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

REVISION = "8d6557168168f91ea6697a4a49ed2ef622b53adb"
FILENAME = "Ternary-Bonsai-2-4B-PTQ1_0.gguf"
SIZE = 949_549_920
SHA256 = "56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60"
URL = f"https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/resolve/{REVISION}/{FILENAME}"


def verified(path):
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="destination model directory")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / FILENAME
    if target.exists():
        if not verified(target):
            parser.error(f"{target} exists but does not match the pinned model; move it first")
        print(f"Verified {target}")
        return
    # Exclusive creation avoids clobbering another concurrent download.
    partial = target.with_suffix(".gguf.part")
    with partial.open("xb") as output:
        try:
            with urllib.request.urlopen(URL, timeout=120) as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    output.write(chunk)
            output.close()
            if not verified(partial):
                raise RuntimeError("download size or SHA-256 mismatch")
            partial.replace(target)
        except BaseException:
            # Windows cannot unlink a file while this handle is open.
            output.close()
            partial.unlink(missing_ok=True)
            raise
    print(f"Verified {target} ({SIZE:,} bytes)")


if __name__ == "__main__":
    main()
