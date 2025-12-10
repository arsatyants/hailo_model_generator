#!/usr/bin/env python3
"""
Python NMS for YOLOv8 Hailo-8 outputs (raw, no NMS)

Usage:
    python3 nms_postprocess.py --input predictions.npy --output detections.npy [--conf 0.25 --iou 0.45]

- Input: numpy file with YOLOv8 raw outputs (shape: [N, 6, 8400])
- Output: numpy file with filtered detections (boxes, scores, class_ids)

This script can be used standalone or imported as a module.
"""
import numpy as np
import argparse
import cv2


def nms_yolov8(boxes, scores, class_ids, conf_thresh=0.25, iou_thresh=0.45):
    """Apply NMS to YOLOv8 predictions (single image)"""
    # Filter by confidence
    conf_mask = scores > conf_thresh
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    class_ids = class_ids[conf_mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # NMS per class
    keep_indices = []

    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_indices = np.where(class_mask)[0]

        indices = cv2.dnn.NMSBoxes(
            class_boxes.tolist(),
            class_scores.tolist(),
            conf_thresh,
            iou_thresh
        )
        # NMSBoxes returns a list of lists or empty list
        if indices is not None and len(indices) > 0:
            # Flatten indices to 1D list
            flat = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
            keep_indices.extend(class_indices[flat].tolist())

    if len(keep_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]


def decode_yolov8_raw(raw_output):
    """
    Decode raw YOLOv8 output (shape: [6, 8400]) to boxes, scores, class_ids
    Assumes output order: [x, y, w, h, conf, class]
    """
    # Transpose to [8400, 6]
    pred = raw_output.transpose(1, 0)
    # Boxes: xywh to xyxy
    x, y, w, h, conf, cls = pred.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = conf
    class_ids = cls.astype(int)
    return boxes, scores, class_ids


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Python NMS post-processing")
    parser.add_argument('--input', required=True, help='Input .npy file (raw YOLOv8 output)')
    parser.add_argument('--output', required=True, help='Output .npy file (filtered detections)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    args = parser.parse_args()

    raw = np.load(args.input)  # shape: [N, 6, 8400] or [6, 8400]
    if raw.ndim == 3:
        raw = raw[0]  # Take first batch
    boxes, scores, class_ids = decode_yolov8_raw(raw)
    boxes, scores, class_ids = nms_yolov8(boxes, scores, class_ids, args.conf, args.iou)
    detections = {'boxes': boxes, 'scores': scores, 'class_ids': class_ids}

    # Save as .npz for dict output
    np.savez(args.output, boxes=boxes, scores=scores, class_ids=class_ids)
    print(f"✅ Saved NMS-filtered detections to {args.output}.npz")


if __name__ == "__main__":
    main()
