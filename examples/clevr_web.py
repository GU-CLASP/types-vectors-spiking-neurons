"""Serve the CLEVR ground-truth and learned-perception comparison UI."""

import argparse
from pathlib import Path

from spinls.web import serve


def default_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CLEVR_v1.0"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("artifacts/clevr-detector/detector.pt"),
    )
    parser.add_argument(
        "--attribute-checkpoint",
        type=Path,
        default=Path("artifacts/clevr-resnet50-fpn/heads.pt"),
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("../overleaf/spinls-overleaf/naloma.pdf"),
    )
    args = parser.parse_args()
    serve(
        args.data_dir,
        split=args.split,
        host=args.host,
        port=args.port,
        detector_checkpoint=args.detector_checkpoint,
        attribute_checkpoint=args.attribute_checkpoint,
        device=args.device,
        detection_threshold=args.detection_threshold,
        paper_path=args.paper,
    )


if __name__ == "__main__":
    main()
