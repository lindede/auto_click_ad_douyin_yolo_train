#!/usr/bin/env python3
"""
导出训练好的模型为 ONNX 格式（用于 CPU 高速推理，无需 PyTorch 运行时）
用法:
  python export/export_onnx.py --weights runs/cls/cls_v1/weights/best.pt --task cls
  python export/export_onnx.py --weights runs/det/det_v1/weights/best.pt --task det
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from configs.settings import YOLO_IMGSZ


def export_onnx(weights: str, task: str, imgsz: int = YOLO_IMGSZ) -> Path:
    """
    将 .pt 权重导出为 ONNX 格式。

    参数:
        weights : .pt 文件路径
        task    : 'cls' 或 'det'（仅用于日志提示）
        imgsz   : 导出时固定的输入尺寸

    返回:
        导出的 .onnx 文件路径
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[错误] 权重文件不存在: {weights_path}")
        sys.exit(1)

    print(f"[导出] 权重文件: {weights_path}")
    print(f"[导出] 任务类型: {task}  输入尺寸: {imgsz}×{imgsz}")
    print("-" * 60)

    model = YOLO(str(weights_path))

    onnx_path = model.export(
        format    = 'onnx',
        imgsz     = imgsz,
        dynamic   = False,   # 固定输入尺寸，推理更快
        simplify  = True,    # onnxsim 简化图结构
        opset     = 17,      # onnxruntime 1.17+ 支持
        half      = False,   # CPU 不使用 FP16
    )

    onnx_path = Path(onnx_path)
    size_mb   = onnx_path.stat().st_size / 1024 / 1024
    print(f"\n✓ 导出成功")
    print(f"  ONNX 文件: {onnx_path}")
    print(f"  文件大小:  {size_mb:.2f} MB")
    print(f"\n  用法（推理）:")
    print(f"    from inference.predictor import Predictor")
    if task == 'cls':
        print(f"    predictor = Predictor(cls_model='{onnx_path}', det_model='...')")
    else:
        print(f"    predictor = Predictor(cls_model='...', det_model='{onnx_path}')")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='导出 YOLO 模型为 ONNX 格式')
    parser.add_argument('--weights', type=str, required=True,
                        help='训练好的 .pt 权重文件路径')
    parser.add_argument('--task',   type=str, required=True,
                        choices=['cls', 'det'],
                        help='模型任务类型：cls=分类  det=检测')
    parser.add_argument('--imgsz',  type=int, default=YOLO_IMGSZ,
                        help=f'推理输入尺寸（默认: {YOLO_IMGSZ}）')
    args = parser.parse_args()

    export_onnx(args.weights, args.task, args.imgsz)


if __name__ == '__main__':
    main()

