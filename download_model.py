"""Helper script for placing the SCRFD 2.5G model.

Because model-host URLs can change, this script accepts a URL instead of hard-coding one.
Recommended model name:
    scrfd_2.5g_bnkps.onnx

Example:
    python download_model.py --url "PASTE_DIRECT_ONNX_URL_HERE"

You can also manually download the ONNX model and place it here:
    models/scrfd_2.5g_bnkps.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Direct URL to SCRFD 2.5G BNKPS ONNX file")
    parser.add_argument("--out", default="models/scrfd_2.5g_bnkps.onnx")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {out} ...")
    urlretrieve(args.url, out)
    print("Done.")


if __name__ == "__main__":
    main()
