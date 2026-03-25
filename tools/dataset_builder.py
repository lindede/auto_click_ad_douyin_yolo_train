#!/usr/bin/env python3
"""
数据集构建工具
用法:
  python tools/dataset_builder.py --task cls   # 构建分类数据集
  python tools/dataset_builder.py --task det   # 初始化检测数据集目录结构
  python tools/dataset_builder.py --task all   # 两者都做（默认）
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.labels import PageClass, DetClass
from configs.settings import (CLS_MANIFEST, CLS_DIR, DET_DIR,
                               TRAIN_RATIO, ROOT)


# ── 分类数据集构建 ─────────────────────────────────────────────
def build_cls_dataset(seed: int = 42) -> None:
    """读取 cls_annotations.json，按 train/val 划分复制到 CLS_DIR。"""
    if not CLS_MANIFEST.exists():
        print(f"[错误] 未找到标注文件: {CLS_MANIFEST}")
        print("  → 请先运行 tools/cls_annotator.py 完成标注")
        return

    ann: dict = json.loads(CLS_MANIFEST.read_text(encoding='utf-8'))
    if not ann:
        print("[错误] 标注文件为空")
        return

    # 按类别分组
    by_class: dict[str, list[Path]] = {c.name: [] for c in PageClass}
    skipped = 0
    for img_path, cls_name in ann.items():
        if cls_name in by_class:
            by_class[cls_name].append(Path(img_path))
        else:
            skipped += 1
    if skipped:
        print(f"[警告] 跳过 {skipped} 条无效标注（标签名不在 PageClass 中）")

    # 清空并重建目录
    if CLS_DIR.exists():
        shutil.rmtree(CLS_DIR)

    random.seed(seed)
    total_train = total_val = 0

    print("\n分类数据集划分结果:")
    print(f"  {'类别':<40} {'train':>6} {'val':>6}")
    print(f"  {'-'*54}")

    for cls_name, paths in by_class.items():
        if not paths:
            continue
        random.shuffle(paths)
        n_train = max(1, int(len(paths) * TRAIN_RATIO))
        train_paths, val_paths = paths[:n_train], paths[n_train:]

        for split, split_paths in [('train', train_paths), ('val', val_paths)]:
            dst = CLS_DIR / split / cls_name
            dst.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                if src.exists():
                    shutil.copy2(src, dst / src.name)

        total_train += len(train_paths)
        total_val   += len(val_paths)
        print(f"  {cls_name:<40} {len(train_paths):>6} {len(val_paths):>6}")

    print(f"  {'-'*54}")
    print(f"  {'合计':<40} {total_train:>6} {total_val:>6}")

    # 生成 YAML 配置
    yaml_path = ROOT / 'configs' / 'cls_dataset.yaml'
    cfg = {
        'path':  str(CLS_DIR),
        'train': 'train',
        'val':   'val',
        'nc':    len(PageClass),
        'names': PageClass.names(),
    }
    yaml_path.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
                         encoding='utf-8')
    print(f"\n✓ 分类数据集已生成: {CLS_DIR}")
    print(f"✓ 配置文件已生成:   {yaml_path}")


# ── 检测数据集目录初始化 ───────────────────────────────────────
def build_det_dataset() -> None:
    """创建检测数据集目录结构并生成 YAML 配置。"""
    for split in ('train', 'val'):
        (DET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    yaml_path = ROOT / 'configs' / 'det_dataset.yaml'
    cfg = {
        'path':  str(DET_DIR),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    len(DetClass),
        'names': DetClass.names(),
    }
    yaml_path.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
                         encoding='utf-8')
    print(f"✓ 检测数据集目录已创建: {DET_DIR}")
    print(f"  结构: images/train  images/val  labels/train  labels/val")
    print(f"✓ 配置文件已生成: {yaml_path}")
    print(f"  → 请将图片放入 images/train(val)，标注文件(.txt)放入 labels/train(val)")


# ── 入口 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='数据集构建工具')
    parser.add_argument('--task', choices=['cls', 'det', 'all'], default='all',
                        help='构建任务类型（默认: all）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    if args.task in ('cls', 'all'):
        print("=" * 56)
        print("  构建分类数据集")
        print("=" * 56)
        build_cls_dataset(seed=args.seed)

    if args.task in ('det', 'all'):
        print("\n" + "=" * 56)
        print("  初始化检测数据集目录")
        print("=" * 56)
        build_det_dataset()


if __name__ == '__main__':
    main()

