#!/usr/bin/env python3
"""
训练目标检测模型 (yolo11n)
用法:
  python train/train_det.py
  python train/train_det.py --epochs 150 --batch 8 --name det_v2
  python train/train_det.py --resume
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from configs.settings import (DET_MODEL_BASE, YOLO_IMGSZ,
                               EPOCHS_DET, BATCH_SIZE, PATIENCE, WORKERS,
                               IMG_WIDTH, IMG_HEIGHT)


def train(
    data:    str  = str(ROOT / 'configs' / 'det_dataset.yaml'),
    epochs:  int  = EPOCHS_DET,
    imgsz:   int  = YOLO_IMGSZ,
    batch:   int  = BATCH_SIZE,
    name:    str  = 'det_v1',
    resume:  bool = False,
    device:  str  = 'cpu',
) -> None:
    """
    训练目标检测模型并保存结果到 runs/det/<name>/。

    参数:
        data   : det_dataset.yaml 路径
        epochs : 训练轮数
        imgsz  : 输入图片尺寸
        batch  : 批大小
        name   : 实验名称
        resume : 是否从上次中断处继续
        device : 'cpu' 或 GPU编号
    """
    print(f"[检测训练] 数据集: {data}")
    print(f"[检测训练] 模型:   {DET_MODEL_BASE}  →  runs/det/{name}")
    print(f"[检测训练] 图片原始分辨率: {IMG_WIDTH}×{IMG_HEIGHT}  训练imgsz={imgsz}")
    print(f"[检测训练] 参数:   epochs={epochs}  batch={batch}  device={device}")
    print("-" * 60)

    model = YOLO(DET_MODEL_BASE)

    results = model.train(
        data      = data,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        project   = str(ROOT / 'runs' / 'det'),
        name      = name,
        resume    = resume,
        rect      = True,    # 矩形训练：保持宽高比，减少填充，更适合竖屏截图
        device    = device,
        workers   = WORKERS,
        patience  = PATIENCE,
        save      = True,
        plots     = True,
        # 针对UI截图的增强策略（禁用破坏UI结构的增强）
        degrees   = 0.0,     # 不旋转
        fliplr    = 0.0,     # 不水平翻转
        flipud    = 0.0,     # 不垂直翻转
        mosaic    = 0.5,     # 适度mosaic增强
        mixup     = 0.0,     # 不mixup（UI元素叠加无意义）
        copy_paste= 0.0,
        scale     = 0.3,     # 适度缩放
        translate = 0.1,
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.4,
        # 检测超参数
        conf      = 0.001,   # 训练时低阈值（NMS后过滤）
        iou       = 0.6,
        box       = 7.5,     # box loss权重
        cls       = 0.5,     # cls loss权重
    )

    best = ROOT / 'runs' / 'det' / name / 'weights' / 'best.pt'
    print(f"\n✓ 训练完成")
    print(f"  最佳权重: {best}")
    print(f"  下一步:   python export/export_onnx.py --weights {best} --task det")
    print(f"  半自动标注: python tools/det_review_tool.py --model {best}")


def main():
    parser = argparse.ArgumentParser(description='训练目标检测模型 (yolo11n)')
    parser.add_argument('--data',   type=str,  default=str(ROOT / 'configs' / 'det_dataset.yaml'))
    parser.add_argument('--epochs', type=int,  default=EPOCHS_DET)
    parser.add_argument('--imgsz',  type=int,  default=YOLO_IMGSZ)
    parser.add_argument('--batch',  type=int,  default=BATCH_SIZE)
    parser.add_argument('--name',   type=str,  default='det_v1')
    parser.add_argument('--device', type=str,  default='cpu', help="'cpu' 或 GPU编号如 '0'")
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
          batch=args.batch, name=args.name, resume=args.resume, device=args.device)


if __name__ == '__main__':
    main()

