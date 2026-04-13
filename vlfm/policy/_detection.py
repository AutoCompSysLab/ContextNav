# Detection mixin for BaseObjectNavPolicy

from typing import Optional

import numpy as np
import torch

from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import ObjectDetections


class DetectionMixin:
    """Object detection logic: hybrid YOLO + GroundingDINO detection and merging."""

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        """
        Hybrid detection:
          (1) target in COCO -> YOLOv7 first (target=0.5, other COCO=0.8).
          (2) target not in COCO -> GroundingDINO always for [target + non-COCO]; YOLOv7 for COCO ctx;
              when overlapped (same class, IoU>0.5), prefer YOLO for non-target classes.
        """
        target = self._target_object
        ctx_all = list(self._context_categories or [])
        ctx_coco = [c for c in ctx_all if c in COCO_CLASSES]
        if target not in COCO_CLASSES:
            ctx_non = [c for c in ctx_all if c not in COCO_CLASSES] + [target]
        else:
            ctx_non = ctx_all
        is_target_coco = target in COCO_CLASSES

        # --- Branch (1): target is COCO ---
        if is_target_coco:
            yolo_classes = [target] + ctx_coco
            yolo = self._coco_object_detector.predict(img)
            yolo.filter_by_class(yolo_classes)
            yolo.filter_by_conf(float(0.8))
            if len(ctx_non) > 0:
                gdino = self._object_detector.predict(img, caption=self._non_coco_caption)
                gdino.filter_by_class(ctx_non)
                gdino.filter_by_conf(float(0.45))
            else:
                return yolo
            if gdino is None or gdino.num_detections == 0:
                return yolo
            if yolo.num_detections == 0:
                return gdino
            return self._merge_detections_prefer_yolo(img, yolo, gdino, iou_th=0.5)

        # --- Branch (2): target is non-COCO ---
        gdino = self._object_detector.predict(img, caption=self._non_coco_caption_with_target)
        gdino.filter_by_class(ctx_non)
        gdino.filter_by_conf(float(0.45))
        if len(ctx_coco) == 0:
            return gdino
        yolo = self._coco_object_detector.predict(img)
        yolo.filter_by_class(ctx_coco)
        yolo.filter_by_conf(float(0.6))
        if yolo.num_detections == 0:
            return gdino
        if gdino.num_detections == 0:
            return yolo

        # Merge, prefer YOLO for non-target overlaps
        if gdino.num_detections > 0 and yolo.num_detections > 0:
            merged = self._merge_detections_prefer_yolo(img, yolo, gdino, prefer_for=set(ctx_coco), exclude=set([target]))
            return merged

    def _merge_detections_prefer_yolo(
        self,
        img: np.ndarray,
        yolo: ObjectDetections,
        gdino: ObjectDetections,
        prefer_for: Optional[set] = None,
        exclude: Optional[set] = None,
        iou_th: float = 0.5,
    ) -> ObjectDetections:
        """
        Merge results from two models on the same image.
        Rule: If a GDINO box overlaps a YOLO box with pixel-coordinate IoU >= iou_th,
        discard GDINO and keep YOLO. Non-overlapping boxes are all kept.
        """
        H, W = img.shape[:2]

        def _scores_1d(logits: torch.Tensor) -> torch.Tensor:
            s = logits if isinstance(logits, torch.Tensor) else torch.tensor(logits, dtype=torch.float32)
            return s.max(dim=1).values if s.ndim == 2 else s.float()

        def _to_pixel_xyxy(det: ObjectDetections) -> np.ndarray:
            bx = det.boxes.detach().cpu().float().numpy() if isinstance(det.boxes, torch.Tensor) else np.asarray(det.boxes, dtype=np.float32)
            if bx.size == 0:
                return bx.reshape(0, 4)
            if float(np.max(bx)) <= 1.5:
                bx = bx * np.array([W, H, W, H], dtype=np.float32)
            x1 = np.minimum(bx[:, 0], bx[:, 2])
            y1 = np.minimum(bx[:, 1], bx[:, 3])
            x2 = np.maximum(bx[:, 0], bx[:, 2])
            y2 = np.maximum(bx[:, 1], bx[:, 3])
            return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        def _iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            if len(A) == 0 or len(B) == 0:
                return np.zeros((len(A), len(B)), dtype=np.float32)
            inter_x1 = np.maximum(A[:, None, 0], B[None, :, 0])
            inter_y1 = np.maximum(A[:, None, 1], B[None, :, 1])
            inter_x2 = np.minimum(A[:, None, 2], B[None, :, 2])
            inter_y2 = np.minimum(A[:, None, 3], B[None, :, 3])
            inter_w = np.clip(inter_x2 - inter_x1, 0, None)
            inter_h = np.clip(inter_y2 - inter_y1, 0, None)
            inter = inter_w * inter_h
            areaA = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
            areaB = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
            union = areaA[:, None] + areaB[None, :] - inter + 1e-6
            return inter / union

        # Quick returns
        if yolo.num_detections == 0 and gdino.num_detections == 0:
            return ObjectDetections(torch.zeros((0, 4), dtype=torch.float32),
                                    torch.zeros((0,), dtype=torch.float32), [], image_source=img, fmt="xyxy")
        if yolo.num_detections == 0:
            return gdino
        if gdino.num_detections == 0:
            return yolo

        y_boxes_pix = _to_pixel_xyxy(yolo)
        g_boxes_pix = _to_pixel_xyxy(gdino)
        y_scores = _scores_1d(yolo.logits).detach().cpu()
        g_scores = _scores_1d(gdino.logits).detach().cpu()
        y_phrases = list(yolo.phrases)
        g_phrases = list(gdino.phrases)

        iou = _iou_matrix(y_boxes_pix, g_boxes_pix)
        g_overlaps = (iou >= float(iou_th)).any(axis=0)
        keep_g_idx = np.where(~g_overlaps)[0]

        merged_boxes_pix = np.vstack([y_boxes_pix, g_boxes_pix[keep_g_idx]]) if len(keep_g_idx) > 0 else y_boxes_pix
        merged_boxes_norm = merged_boxes_pix / np.array([W, H, W, H], dtype=np.float32)
        merged_logits = torch.cat([y_scores, g_scores[torch.tensor(keep_g_idx, dtype=torch.long)]], dim=0) if len(keep_g_idx) > 0 else y_scores
        merged_phrases = y_phrases + [g_phrases[i] for i in keep_g_idx]

        return ObjectDetections(
            boxes=torch.from_numpy(merged_boxes_norm.astype(np.float32)),
            logits=merged_logits.float(),
            phrases=merged_phrases,
            image_source=img,
            fmt="xyxy",
        )
