#!/usr/bin/env python3
"""
训练页面分类模型 (yolo11n-cls)
用法:
  python train/train_cls.py
  python train/train_cls.py --epochs 150 --batch 8 --name cls_v2
  python train/train_cls.py --resume  # 从上次中断处继续
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from configs.settings import (CLS_MODEL_BASE, YOLO_IMGSZ,
                               EPOCHS_CLS, BATCH_SIZE, PATIENCE, WORKERS, CLS_DIR)


def train(
    data:    str  = str(CLS_DIR),   # 分类任务传目录，不传 yaml
    epochs:  int  = EPOCHS_CLS,
    imgsz:   int  = YOLO_IMGSZ,
    batch:   int  = BATCH_SIZE,
    name:    str  = 'cls_v1',
    resume:  bool = False,
    device:  str  = 'cpu',
) -> None:
    """
    训练分类模型并保存结果到 runs/cls/<name>/。

    参数:
        data   : cls_dataset.yaml 路径
        epochs : 训练轮数
        imgsz  : 输入图片尺寸（会等比缩放+letterbox）
        batch  : 批大小
        name   : 实验名称（结果保存在 runs/cls/<name>/）
        resume : 是否从上次中断处继续训练
        device : 'cpu' 或 '0'（GPU编号）
    """
    print(f"[分类训练] 数据集: {data}")
    print(f"[分类训练] 模型:   {CLS_MODEL_BASE}  →  runs/cls/{name}")
    print(f"[分类训练] 参数:   epochs={epochs}  imgsz={imgsz}  batch={batch}  device={device}")
    print("-" * 60)

    model = YOLO(CLS_MODEL_BASE)

    results = model.train(
        data      = data,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        project   = str(ROOT / 'runs' / 'cls'),
        name      = name,
        resume    = resume,
        device    = device,
        workers   = WORKERS,
        patience  = PATIENCE,
        save      = True,
        plots     = True,
        # 针对小分辨率截图的增强策略
        degrees   = 0.0,    # 不旋转（手机截图方向固定）
        fliplr    = 0.0,    # 不水平翻转（UI方向固定）
        flipud    = 0.0,    # 不垂直翻转
        scale     = 0.2,    # 轻微缩放扰动
        translate = 0.1,    # 轻微平移扰动
        hsv_h     = 0.015,  # 色调微调
        hsv_s     = 0.4,    # 饱和度微调
        hsv_v     = 0.4,    # 亮度微调
    )

    best = ROOT / 'runs' / 'cls' / name / 'weights' / 'best.pt'
    print(f"\n✓ 训练完成")
    print(f"  最佳权重: {best}")
    print(f"  下一步:   python export/export_onnx.py --weights {best} --task cls")


def main():
    parser = argparse.ArgumentParser(description='训练页面分类模型 (yolo11n-cls)')
    parser.add_argument('--data',   type=str,  default=str(CLS_DIR),
                        help='分类数据集目录（含 train/val 子文件夹，默认: datasets/classification）')
    parser.add_argument('--epochs', type=int,  default=EPOCHS_CLS)
    parser.add_argument('--imgsz',  type=int,  default=YOLO_IMGSZ)
    parser.add_argument('--batch',  type=int,  default=BATCH_SIZE)
    parser.add_argument('--name',   type=str,  default='cls_v1')
    parser.add_argument('--device', type=str,  default='cpu', help="'cpu' 或 GPU编号如 '0'")
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续训练')
    args = parser.parse_args()

    train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
          batch=args.batch, name=args.name, resume=args.resume, device=args.device)


if __name__ == '__main__':
    main()

