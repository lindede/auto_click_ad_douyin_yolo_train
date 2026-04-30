#!/usr/bin/env python3
"""
清理 cls_annotations.json：移除图片文件已不存在的条目。

用法:
  python tools/clean_cls_annotations.py              # 默认处理 datasets/cls_annotations.json
  python tools/clean_cls_annotations.py --dry-run      # 只打印将删除的条目，不写回文件
  python tools/clean_cls_annotations.py --json PATH  # 指定其它 JSON 路径
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from configs.settings import ROOT

DEFAULT_CLS_JSON = Path("datasets/cls_annotations.json")


def resolve_image_path(key: str) -> Path:
    """将 JSON 中的路径键解析为实际文件路径（相对路径相对项目根）。"""
    p = Path(key)
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def clean_manifest(manifest_path: Path, *, dry_run: bool) -> int:
    """
    返回删除的条目数；dry_run 时不写回文件。
    """
    if not manifest_path.exists():
        print(f"[错误] 文件不存在: {manifest_path}")
        return -1

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("[错误] JSON 根节点必须是对象（路径 -> 类别名）")
        return -1

    missing: list[str] = []
    kept: dict[str, str] = {}
    for img_key, cls_name in data.items():
        resolved = resolve_image_path(str(img_key))
        if resolved.is_file():
            kept[str(img_key)] = cls_name
        else:
            missing.append(str(img_key))

    n = len(missing)
    if n:
        print(f"将移除 {n} 条（文件不存在）:")
        for k in missing[:50]:
            print(f"  - {k}")
        if n > 50:
            print(f"  ... 另有 {n - 50} 条")
    else:
        print("无需清理：所有条目对应的文件均存在。")

    if dry_run:
        print(f"[dry-run] 保留 {len(kept)} 条，未修改文件。")
        return n

    manifest_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写回 {manifest_path}，保留 {len(kept)} 条。")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="清理 cls_annotations.json 中无效图片路径")
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_CLS_JSON,
        help="标注 JSON 路径（默认: datasets/cls_annotations.json）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅报告将删除的条目，不写回文件",
    )
    args = parser.parse_args()
    path = args.json
    if not path.is_absolute():
        path = (_PKG_ROOT / path).resolve()

    clean_manifest(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
