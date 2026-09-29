"""Serve the minimal CLEVR ground-truth browser UI."""

import argparse
from pathlib import Path

from spinls.web import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CLEVR_v1.0"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.data_dir, split=args.split, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
