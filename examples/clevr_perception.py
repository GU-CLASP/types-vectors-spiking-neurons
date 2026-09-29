"""Offline perceptual matching demo; optional real Faster R-CNN architecture smoke test."""
import argparse

from spinls.perception import Detection, build_untrained_faster_rcnn, model_from_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-smoke", action="store_true",
                        help="Run random-weight Faster R-CNN on a tiny blank image (no downloads)")
    args = parser.parse_args()
    if args.architecture_smoke:
        import torch
        from spinls.perception import FasterRCNNDetector
        torch.manual_seed(0)
        detector = FasterRCNNDetector(
            build_untrained_faster_rcnn(min_size=64, max_size=64,
                                       rpn_pre_nms_top_n_test=20, rpn_post_nms_top_n_test=10,
                                       box_detections_per_img=5),
            attribute_classifier=lambda crop: {}, score_threshold=0.0,
        )
        output = detector.detect(torch.zeros(3, 64, 64))
        print(f"Architecture smoke: {len(output)} random detections; no learned CLEVR attributes.")

    # A fixture at the detector/attribute-classifier boundary, not image inference.
    detections = [
        Detection("demo-0", (5, 5, 25, 25), 0.98,
                  {"color": {"red": 0.95, "blue": 0.05}, "shape": {"cube": 1.0},
                   "size": {"small": 0.9, "large": 0.1}, "material": {"rubber": 1.0}}),
        Detection("demo-1", (35, 5, 60, 30), 0.97,
                  {"color": {"blue": 1.0}, "shape": {"sphere": 1.0}}),
    ]
    model = model_from_detections(detections, attribute_threshold=0.7)
    # Controlled descriptions are explicit compositions, not a natural-language parser.
    for text, attributes in [
        ("a small red cube", {"size": "small", "color": "red", "shape": "cube"}),
        ("a blue cube", {"color": "blue", "shape": "cube"}),
        ("a blue sphere", {"color": "blue", "shape": "sphere"}),
    ]:
        print(f"{text}: {model.matches(**attributes)}")


if __name__ == "__main__":
    main()
