#!/usr/bin/env python3
"""
两阶段推理模块
  第一阶段: 页面分类 (yolo11n-cls / ONNX)
  第二阶段: 目标检测 (yolo11n / ONNX)

用法示例:
  from inference.predictor import Predictor
  p = Predictor(
      cls_model='runs/cls/cls_v1/weights/best.onnx',
      det_model='runs/det/det_v1/weights/best.onnx',
  )
  result = p.predict_file('datasets/raw/frame_001.png')
  print(result.page_class, result.detections)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.labels import PageClass, DetClass
from configs.settings import CLS_CONF_THRESH, DET_CONF_THRESH


@dataclass
class Detection:
    """单个目标检测结果"""
    class_id:   int
    class_name: str
    confidence: float
    bbox:       tuple[int, int, int, int]   # (x1, y1, x2, y2) 像素坐标

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"box=({x1},{y1},{x2},{y2}))")


@dataclass
class PredictResult:
    """完整的一次推理结果"""
    page_class:      Optional[PageClass]   # 识别到的页面类型（None=低置信度）
    page_confidence: float                 # 页面分类置信度
    detections:      list[Detection] = field(default_factory=list)

    def __repr__(self) -> str:
        pg = self.page_class.name if self.page_class else 'Unknown'
        return (f"PredictResult(page={pg}({self.page_confidence:.2f}), "
                f"detections={len(self.detections)})")


class Predictor:
    """
    两阶段推理器，支持 .pt 和 .onnx 权重。

    参数:
        cls_model        : 分类模型路径（.pt 或 .onnx）
        det_model        : 检测模型路径（.pt 或 .onnx）
        cls_conf_thresh  : 页面分类置信度阈值（低于此值时 page_class=None）
        det_conf_thresh  : 目标检测置信度阈值
    """

    def __init__(
        self,
        cls_model:       str,
        det_model:       str,
        cls_conf_thresh: float = CLS_CONF_THRESH,
        det_conf_thresh: float = DET_CONF_THRESH,
    ):
        from ultralytics import YOLO
        self.cls_model       = YOLO(cls_model)
        self.det_model       = YOLO(det_model)
        self.cls_conf_thresh = cls_conf_thresh
        self.det_conf_thresh = det_conf_thresh
        print(f"[Predictor] 分类模型: {cls_model}")
        print(f"[Predictor] 检测模型: {det_model}")

    def predict(self, img: np.ndarray) -> PredictResult:
        """
        对一张图片（BGR numpy array）进行两阶段推理。

        返回:
            PredictResult
        """
        # ── 第一阶段: 页面分类 ────────────────────────────────
        cls_out   = self.cls_model.predict(img, verbose=False)[0]
        probs     = cls_out.probs
        cls_idx   = int(probs.top1)
        cls_conf  = float(probs.top1conf)

        page_class = (PageClass.from_index(cls_idx)
                      if cls_conf >= self.cls_conf_thresh and cls_idx < len(PageClass)
                      else None)

        # ── 第二阶段: 目标检测 ────────────────────────────────
        det_out    = self.det_model.predict(img, conf=self.det_conf_thresh,
                                            verbose=False)[0]
        detections = []
        for box in det_out.boxes:
            det_cls_id = int(box.cls[0])
            detections.append(Detection(
                class_id   = det_cls_id,
                class_name = (DetClass.from_index(det_cls_id).name
                              if det_cls_id < len(DetClass) else 'unknown'),
                confidence = float(box.conf[0]),
                bbox       = tuple(map(int, box.xyxy[0].tolist())),
            ))

        return PredictResult(
            page_class      = page_class,
            page_confidence = cls_conf,
            detections      = detections,
        )

    def predict_file(self, img_path: str | Path) -> PredictResult:
        """从文件路径读取图片并推理。"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        return self.predict(img)

