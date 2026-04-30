#!/usr/bin/env python3
"""
检查 datasets/raw 下图片分辨率是否与 configs/settings.py 一致。

用法:
  python tools/check_raw_resolution.py
  python tools/check_raw_resolution.py --src datasets/raw
  python tools/check_raw_resolution.py --show-mismatch 20
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.settings import RAW_DIR, IMG_WIDTH, IMG_HEIGHT

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_raw_resolution(src: Path, show_mismatch: int) -> int:
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[提示] 未在目录中发现图片: {src}")
        return 0

    expected = (IMG_WIDTH, IMG_HEIGHT)
    resolution_counter: Counter[tuple[int, int]] = Counter()
    mismatches: list[tuple[Path, tuple[int, int]]] = []
    failed: list[tuple[Path, str]] = []

    for p in images:
        try:
            with Image.open(p) as im:
                wh = (im.width, im.height)
        except Exception as e:  # pragma: no cover
            failed.append((p, str(e)))
            continue

        resolution_counter[wh] += 1
        if wh != expected:
            mismatches.append((p, wh))

    print(f"[检查目录] {src}")
    print(f"[目标分辨率] {expected[0]}x{expected[1]}")
    print(f"[图片总数] {len(images)}")
    print(f"[可读图片] {sum(resolution_counter.values())}")
    print(f"[不匹配数] {len(mismatches)}")
    print()

    print("[分辨率分布 Top 10]")
    for (w, h), cnt in resolution_counter.most_common(10):
        mark = " (expected)" if (w, h) == expected else ""
        print(f"  - {w}x{h}: {cnt}{mark}")

    if mismatches:
        print()
        print(f"[不匹配样例 Top {show_mismatch}]")
        for p, (w, h) in mismatches[:show_mismatch]:
            print(f"  - {p}: {w}x{h}")

    if failed:
        print()
        print(f"[读取失败] {len(failed)}")
        for p, err in failed[: min(10, len(failed))]:
            print(f"  - {p}: {err}")

    return 1 if mismatches else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 raw 图片分辨率一致性")
    parser.add_argument(
        "--src",
        type=Path,
        default=RAW_DIR,
        help=f"待检查图片目录 (默认: {RAW_DIR})",
    )
    parser.add_argument(
        "--show-mismatch",
        type=int,
        default=20,
        help="输出不匹配样例数量 (默认: 20)",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"[错误] 目录不存在: {args.src}")
        sys.exit(2)

    exit_code = check_raw_resolution(args.src, args.show_mismatch)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
