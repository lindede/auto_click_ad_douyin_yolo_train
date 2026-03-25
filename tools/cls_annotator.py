#!/usr/bin/env python3
"""
快速页面分类标注工具
用法: python tools/cls_annotator.py [--src datasets/raw]

操作说明:
  数字键 1-8  → 给当前图片打标签并自动跳下一张
  ← / →       → 上一张 / 下一张（不标注）
  u           → 撤销当前图片的标注
  q / Esc     → 保存并退出
"""

import argparse
import json
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.labels import PageClass
from configs.settings import CLS_MANIFEST, RAW_DIR, IMG_WIDTH, IMG_HEIGHT

IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DISPLAY_H  = 600                                       # 图片显示高度(px)
DISPLAY_W  = int(DISPLAY_H * IMG_WIDTH / IMG_HEIGHT)   # 保持宽高比

# ── 配色（Catppuccin Mocha 风格）──────────────────────────────
BG       = '#1e1e2e'
PANEL    = '#313244'
FG       = '#cdd6f4'
ACCENT   = '#89b4fa'
MUTED    = '#6c7086'
SUCCESS  = '#a6e3a1'
CLS_COLORS = ['#f38ba8','#fab387','#f9e2af','#a6e3a1',
              '#94e2d5','#89dceb','#89b4fa','#b4befe']


class ClsAnnotator:
    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.images  = sorted(p for p in src_dir.rglob('*') if p.suffix.lower() in IMG_EXTS)
        if not self.images:
            raise FileNotFoundError(f"在 {src_dir} 中未找到任何图片")

        # 加载已有标注
        self.ann: dict[str, str] = {}
        if CLS_MANIFEST.exists():
            self.ann = json.loads(CLS_MANIFEST.read_text(encoding='utf-8'))

        # 定位到第一张未标注图片
        self.idx = next((i for i, p in enumerate(self.images)
                         if str(p) not in self.ann), 0)
        self._build_ui()

    # ── UI 构建 ────────────────────────────────────────────────
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("页面分类标注工具")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # 顶部：进度信息
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=12, pady=(8, 2))
        self.lbl_prog = tk.Label(top, bg=BG, fg=FG, font=('Consolas', 11, 'bold'))
        self.lbl_prog.pack(side='left')
        self.lbl_file = tk.Label(top, bg=BG, fg=MUTED, font=('Consolas', 9))
        self.lbl_file.pack(side='right')

        # 进度条
        self.prog_var = tk.DoubleVar()
        ttk.Progressbar(self.root, variable=self.prog_var,
                        maximum=len(self.images)).pack(fill='x', padx=12, pady=(2, 6))

        # 主体：图片 + 右侧面板
        body = tk.Frame(self.root, bg=BG)
        body.pack(padx=12, pady=4)

        self.canvas = tk.Canvas(body, width=DISPLAY_W, height=DISPLAY_H,
                                bg='#11111b', highlightthickness=1,
                                highlightbackground=PANEL)
        self.canvas.pack(side='left', padx=(0, 14))

        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='y')

        tk.Label(right, text='快捷键选择分类:', bg=BG, fg=ACCENT,
                 font=('Consolas', 10, 'bold')).pack(anchor='w', pady=(0, 6))

        self.cls_btns: list[tk.Button] = []
        for i, cls in enumerate(PageClass):
            color = CLS_COLORS[i]
            btn = tk.Button(right, text=f'[{i+1}]  {cls.name}',
                            bg=PANEL, fg=color, font=('Consolas', 10),
                            activebackground='#45475a', activeforeground=color,
                            relief='flat', anchor='w', padx=10, pady=5, width=30,
                            command=lambda c=cls: self._label(c))
            btn.pack(fill='x', pady=1)
            self.cls_btns.append(btn)

        tk.Frame(right, bg=BG, height=10).pack()
        tk.Label(right, text='当前标注:', bg=BG, fg=ACCENT,
                 font=('Consolas', 10, 'bold')).pack(anchor='w')
        self.lbl_cur = tk.Label(right, text='(未标注)', bg=BG, fg=MUTED,
                                font=('Consolas', 11, 'bold'))
        self.lbl_cur.pack(anchor='w')

        # 底部导航栏
        nav = tk.Frame(self.root, bg=BG)
        nav.pack(pady=8)
        for txt, cmd in [('←  上一张', lambda: self._nav(-1)),
                         ('下一张  →', lambda: self._nav(1)),
                         ('u  撤销',   self._undo),
                         ('q  退出',   self._quit)]:
            tk.Button(nav, text=txt, command=cmd, bg=PANEL, fg=FG,
                      activebackground='#45475a', relief='flat',
                      padx=12, pady=5, font=('Consolas', 10)).pack(side='left', padx=4)

        self._bind_keys()
        self._refresh()
        self.root.mainloop()

    def _bind_keys(self):
        for i, cls in enumerate(PageClass, 1):
            self.root.bind(str(i), lambda e, c=cls: self._label(c))
        self.root.bind('<Left>',  lambda e: self._nav(-1))
        self.root.bind('<Right>', lambda e: self._nav(1))
        self.root.bind('u', lambda e: self._undo())
        self.root.bind('q', lambda e: self._quit())
        self.root.bind('<Escape>', lambda e: self._quit())

    # ── 操作 ───────────────────────────────────────────────────
    def _label(self, cls: PageClass):
        self.ann[str(self.images[self.idx])] = cls.name
        self._save()
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._refresh()
        else:
            self._refresh()
            messagebox.showinfo("完成 🎉", "所有图片已标注完成！")

    def _nav(self, delta: int):
        new = self.idx + delta
        if 0 <= new < len(self.images):
            self.idx = new
            self._refresh()

    def _undo(self):
        key = str(self.images[self.idx])
        if key in self.ann:
            del self.ann[key]
            self._save()
            self._refresh()

    def _quit(self):
        self._save()
        self.root.destroy()

    def _save(self):
        CLS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        CLS_MANIFEST.write_text(
            json.dumps(self.ann, ensure_ascii=False, indent=2), encoding='utf-8')

    # ── 刷新显示 ───────────────────────────────────────────────
    def _refresh(self):
        img = Image.open(self.images[self.idx]).resize(
            (DISPLAY_W, DISPLAY_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

        done  = len(self.ann)
        total = len(self.images)
        self.lbl_prog.config(text=f'进度 {done}/{total}  |  当前第 {self.idx+1} 张')
        self.lbl_file.config(text=self.images[self.idx].name)
        self.prog_var.set(done)

        cur = self.ann.get(str(self.images[self.idx]), '')
        if cur:
            self.lbl_cur.config(text=f'✓  {cur}', fg=SUCCESS)
        else:
            self.lbl_cur.config(text='(未标注)', fg=MUTED)

        for i, btn in enumerate(self.cls_btns):
            matched = list(PageClass)[i].name == cur
            btn.config(bg='#1e3a5f' if matched else PANEL)


def main():
    parser = argparse.ArgumentParser(description='页面分类标注工具')
    parser.add_argument('--src', type=Path, default=RAW_DIR,
                        help=f'原始图片目录 (默认: {RAW_DIR})')
    args = parser.parse_args()
    if not args.src.exists():
        print(f"[错误] 目录不存在: {args.src}")
        sys.exit(1)
    ClsAnnotator(args.src)


if __name__ == '__main__':
    main()

