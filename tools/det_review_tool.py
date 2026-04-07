#!/usr/bin/env python3
"""
半自动检测标注审核工具
  - 无模型模式: 纯手工画框标注
  - 有模型模式: 模型预生成框 → 人工审核/修改/补充

用法:
  # 纯手工模式（冷启动）
  python tools/det_review_tool.py --img_dir datasets/detection/images/train

  # 半自动模式（已有初始检测模型）
  python tools/det_review_tool.py --img_dir datasets/detection/images/train \\
                                   --model   runs/det/det_v1/weights/best.pt

操作说明:
  鼠标左键拖拽          → 画新的检测框
  左键单击已有框         → 选中（高亮显示）
  右键单击已有框         → 删除该框
  选中框后按数字键 1-N   → 修改类别
  选中框后按 Delete/d   → 删除选中框
  N / →                 → 保存当前图片标注并跳到下一张
  P / ←                 → 保存并跳到上一张
  S                     → 仅保存（不切换）
  Q / Esc               → 保存并退出
"""

import argparse
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.labels import DetClass
from configs.settings import DET_DIR, IMG_WIDTH, IMG_HEIGHT, DATASETS_DIR

IMG_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DISPLAY_H = 640
DISPLAY_W = int(DISPLAY_H * IMG_WIDTH / IMG_HEIGHT)   # ≈289 px

BG        = '#1e1e2e'
PANEL     = '#313244'
FG        = '#cdd6f4'
ACCENT    = '#89b4fa'
MUTED     = '#6c7086'
SEL_COLOR = '#f9e2af'    # 选中框颜色

# 每个类别对应颜色（循环使用）
CLS_COLORS = [
    '#f38ba8','#fab387','#f9e2af','#a6e3a1','#94e2d5',
    '#89dceb','#89b4fa','#b4befe','#cba6f7','#eba0ac',
    '#f2cdcd','#e8c7a0','#d9e0ee','#c9cbff',
]


def cls_color(cls_id: int) -> str:
    return CLS_COLORS[cls_id % len(CLS_COLORS)]


class BBox:
    """YOLO格式的标注框（坐标以原始图像像素为单位）"""
    def __init__(self, cls_id: int, x1: float, y1: float, x2: float, y2: float):
        self.cls_id = cls_id
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def to_yolo_line(self, img_w: int, img_h: int) -> str:
        xc = (self.x1 + self.x2) / 2 / img_w
        yc = (self.y1 + self.y2) / 2 / img_h
        w  = (self.x2 - self.x1) / img_w
        h  = (self.y2 - self.y1) / img_h
        return f"{self.cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def from_yolo_line(line: str, img_w: int, img_h: int) -> "BBox":
        parts  = line.strip().split()
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        x1 = (xc - w / 2) * img_w;  x2 = (xc + w / 2) * img_w
        y1 = (yc - h / 2) * img_h;  y2 = (yc + h / 2) * img_h
        return BBox(cls_id, x1, y1, x2, y2)

    @property
    def cls_name(self) -> str:
        try:
            return DetClass.from_index(self.cls_id).name
        except IndexError:
            return f"cls_{self.cls_id}"


class DetReviewTool:
    def __init__(self, img_dir: Path, label_dir: Path, model_path: str | None):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        label_dir.mkdir(parents=True, exist_ok=True)

        self.images = sorted(p for p in img_dir.rglob('*') if p.suffix.lower() in IMG_EXTS)
        if not self.images:
            raise FileNotFoundError(f"未在 {img_dir} 中找到图片")

        # 加载检测模型（可选）
        self.model = None
        if model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[模型] 已加载: {model_path}")

        self.idx      = 0
        self.boxes:   list[BBox] = []
        self.selected: int | None = None   # 选中框的索引

        # 画框状态
        self._drag_start: tuple[float, float] | None = None
        self._drag_rect_id: int | None = None

        # 缩放比例（canvas坐标 ↔ 图片像素）
        self._img_w = IMG_WIDTH
        self._img_h = IMG_HEIGHT
        self._scale_x = DISPLAY_W / self._img_w
        self._scale_y = DISPLAY_H / self._img_h

        self._build_ui()

    # ── UI 构建 ────────────────────────────────────────────────
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("检测标注审核工具")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # 顶部信息栏
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=10, pady=(8, 2))
        self.lbl_prog = tk.Label(top, bg=BG, fg=FG, font=('Consolas', 11, 'bold'))
        self.lbl_prog.pack(side='left')
        self.lbl_file = tk.Label(top, bg=BG, fg=MUTED, font=('Consolas', 9))
        self.lbl_file.pack(side='right')

        self.prog_var = tk.DoubleVar()
        ttk.Progressbar(self.root, variable=self.prog_var,
                        maximum=len(self.images)).pack(fill='x', padx=10, pady=(2, 4))

        # 主体
        body = tk.Frame(self.root, bg=BG)
        body.pack(padx=10, pady=4)

        # Canvas（图片显示+画框）
        self.canvas = tk.Canvas(body, width=DISPLAY_W, height=DISPLAY_H,
                                bg='#11111b', cursor='crosshair',
                                highlightthickness=1, highlightbackground=PANEL)
        self.canvas.pack(side='left', padx=(0, 12))
        self.canvas.bind('<ButtonPress-1>',   self._on_press)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>',        self._on_right_click)

        # 右侧面板
        right = tk.Frame(body, bg=BG, width=220)
        right.pack(side='left', fill='y')
        right.pack_propagate(False)

        tk.Label(right, text='类别 (数字键修改选中框):', bg=BG, fg=ACCENT,
                 font=('Consolas', 9, 'bold')).pack(anchor='w', pady=(0, 4))

        # 类别列表（可滚动）
        frame_list = tk.Frame(right, bg=BG)
        frame_list.pack(fill='both', expand=True)
        sb = tk.Scrollbar(frame_list, orient='vertical')
        self.cls_listbox = tk.Listbox(frame_list, bg=PANEL, fg=FG, font=('Consolas', 9),
                                      selectbackground='#45475a', relief='flat',
                                      yscrollcommand=sb.set, height=16)
        sb.config(command=self.cls_listbox.yview)
        self.cls_listbox.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        for i, cls in enumerate(DetClass):
            color = cls_color(i)
            self.cls_listbox.insert('end', f' [{i+1:02d}] {cls.name}')
            self.cls_listbox.itemconfig(i, fg=color)
        self.cls_listbox.bind('<<ListboxSelect>>', self._on_cls_select)

        # 当前框信息
        tk.Frame(right, bg=BG, height=8).pack()
        tk.Label(right, text='选中框:', bg=BG, fg=ACCENT,
                 font=('Consolas', 9, 'bold')).pack(anchor='w')
        self.lbl_sel = tk.Label(right, text='(无)', bg=BG, fg=MUTED,
                                font=('Consolas', 10, 'bold'), wraplength=210)
        self.lbl_sel.pack(anchor='w')

        tk.Frame(right, bg=BG, height=4).pack()
        tk.Label(right, text='框数量:', bg=BG, fg=ACCENT,
                 font=('Consolas', 9, 'bold')).pack(anchor='w')
        self.lbl_count = tk.Label(right, text='0', bg=BG, fg=FG, font=('Consolas', 11, 'bold'))
        self.lbl_count.pack(anchor='w')

        # 底部导航
        nav = tk.Frame(self.root, bg=BG)
        nav.pack(pady=6)
        btns = [('← 上一张(P)', lambda: self._navigate(-1)),
                ('保存(S)',      self._save_current),
                ('下一张 →(N)', lambda: self._navigate(1)),
                ('预标注(A)',    self._auto_annotate),
                ('清空(C)',      self._clear_boxes),
                ('退出(Q)',      self._quit)]
        for txt, cmd in btns:
            tk.Button(nav, text=txt, command=cmd, bg=PANEL, fg=FG,
                      activebackground='#45475a', relief='flat',
                      padx=8, pady=4, font=('Consolas', 9)).pack(side='left', padx=3)

        self._bind_keys()
        self._load_image()
        self.root.mainloop()

    def _bind_keys(self):
        for i in range(len(DetClass)):
            key = str(i + 1) if i < 9 else None
            if key:
                self.root.bind(key, lambda e, idx=i: self._set_cls(idx))
        self.root.bind('<Delete>', lambda e: self._delete_selected())
        self.root.bind('d',       lambda e: self._delete_selected())
        self.root.bind('n',       lambda e: self._navigate(1))
        self.root.bind('p',       lambda e: self._navigate(-1))
        self.root.bind('<Right>', lambda e: self._navigate(1))
        self.root.bind('<Left>',  lambda e: self._navigate(-1))
        self.root.bind('s',       lambda e: self._save_current())
        self.root.bind('a',       lambda e: self._auto_annotate())
        self.root.bind('c',       lambda e: self._clear_boxes())
        self.root.bind('q',       lambda e: self._quit())
        self.root.bind('<Escape>',lambda e: self._quit())

    # ── 鼠标交互 ───────────────────────────────────────────────
    def _canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        return cx / self._scale_x, cy / self._scale_y

    def _img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self._scale_x, iy * self._scale_y

    def _on_press(self, event):
        # 先判断是否点中已有框
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self.selected = hit
            self._redraw()
            return
        # 否则开始画新框
        self.selected = None
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
        x0, y0 = self._drag_start
        self._drag_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=SEL_COLOR, width=2, dash=(4, 2))

    def _on_release(self, event):
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start    = None
        self._drag_rect_id  = None

        # 过滤太小的框
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self._redraw()
            return

        ix0, iy0 = self._canvas_to_img(min(x0, x1), min(y0, y1))
        ix1, iy1 = self._canvas_to_img(max(x0, x1), max(y0, y1))
        # 默认用类别0，提示用户选择
        new_box = BBox(0, ix0, iy0, ix1, iy1)
        self.boxes.append(new_box)
        self.selected = len(self.boxes) - 1
        self._redraw()

    def _on_right_click(self, event):
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self.boxes.pop(hit)
            self.selected = None
            self._redraw()

    def _hit_test(self, cx: float, cy: float) -> int | None:
        """返回被点击的框索引，优先最小面积框"""
        ix, iy = self._canvas_to_img(cx, cy)
        candidates = []
        for i, b in enumerate(self.boxes):
            if b.x1 <= ix <= b.x2 and b.y1 <= iy <= b.y2:
                area = (b.x2 - b.x1) * (b.y2 - b.y1)
                candidates.append((area, i))
        return min(candidates, key=lambda t: t[0])[1] if candidates else None

    # ── 类别修改 ───────────────────────────────────────────────
    def _set_cls(self, cls_id: int):
        if self.selected is not None and self.selected < len(self.boxes):
            self.boxes[self.selected].cls_id = cls_id
            self._redraw()

    def _on_cls_select(self, _event):
        sel = self.cls_listbox.curselection()
        if sel:
            self._set_cls(sel[0])

    def _delete_selected(self):
        if self.selected is not None and self.selected < len(self.boxes):
            self.boxes.pop(self.selected)
            self.selected = None
            self._redraw()

    def _clear_boxes(self):
        self.boxes.clear()
        self.selected = None
        self._redraw()

    # ── 半自动预标注 ───────────────────────────────────────────
    def _auto_annotate(self):
        if self.model is None:
            from tkinter import messagebox
            messagebox.showinfo("提示", "请用 --model 参数指定检测模型后启动工具")
            return
        img_path = str(self.images[self.idx])
        results  = self.model.predict(img_path, conf=0.3, verbose=False)[0]
        new_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            new_boxes.append(BBox(cls_id, x1, y1, x2, y2))
        self.boxes   = new_boxes
        self.selected = None
        self._redraw()

    # ── 保存/加载标注 ──────────────────────────────────────────
    def _label_path(self, img_path: Path) -> Path:
        return self.label_dir / (img_path.stem + '.txt')

    def _save_current(self):
        label_file = self._label_path(self.images[self.idx])
        lines = [b.to_yolo_line(self._img_w, self._img_h) for b in self.boxes]
        label_file.write_text('\n'.join(lines), encoding='utf-8')

    def _load_labels(self, img_path: Path):
        self.boxes    = []
        self.selected = None
        lf = self._label_path(img_path)
        if lf.exists():
            for line in lf.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    self.boxes.append(BBox.from_yolo_line(line, self._img_w, self._img_h))

    # ── 图片加载与绘制 ─────────────────────────────────────────
    def _load_image(self):
        img_path = self.images[self.idx]
        img      = Image.open(img_path)
        self._img_w = img.width
        self._img_h = img.height
        self._scale_x = DISPLAY_W / self._img_w
        self._scale_y = DISPLAY_H / self._img_h
        img_disp = img.resize((DISPLAY_W, DISPLAY_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_disp)
        self._load_labels(img_path)
        self._redraw()

    def _redraw(self):
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

        for i, b in enumerate(self.boxes):
            cx1, cy1 = self._img_to_canvas(b.x1, b.y1)
            cx2, cy2 = self._img_to_canvas(b.x2, b.y2)
            color  = cls_color(b.cls_id)
            width  = 3 if i == self.selected else 2
            outline= SEL_COLOR if i == self.selected else color
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                         outline=outline, width=width)
            self.canvas.create_text(cx1 + 2, cy1 + 2, anchor='nw',
                                    text=f"{b.cls_id}:{b.cls_name}",
                                    fill=outline, font=('Consolas', 8, 'bold'))

        # 更新信息栏
        total = len(self.images)
        done  = sum(1 for p in self.images if self._label_path(p).exists())
        self.lbl_prog.config(text=f'进度 {done}/{total}  |  当前第 {self.idx+1} 张')
        self.lbl_file.config(text=self.images[self.idx].name)
        self.prog_var.set(done)
        self.lbl_count.config(text=str(len(self.boxes)))

        if self.selected is not None and self.selected < len(self.boxes):
            b = self.boxes[self.selected]
            self.lbl_sel.config(text=f'✓ [{b.cls_id}] {b.cls_name}', fg=SEL_COLOR)
        else:
            self.lbl_sel.config(text='(无)', fg=MUTED)

    # ── 导航 ───────────────────────────────────────────────────
    def _navigate(self, delta: int):
        self._save_current()
        new = self.idx + delta
        if 0 <= new < len(self.images):
            self.idx = new
            self._load_image()

    def _quit(self):
        self._save_current()
        self.root.destroy()


# ── 入口 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='检测标注审核工具')
    parser.add_argument('--img_dir',   type=Path,
                        default=DATASETS_DIR / 'raw',
                        help='图片目录（默认: datasets/raw）')
    parser.add_argument('--label_dir', type=Path, 
                        default=DATASETS_DIR / 'raw_det_labels',
                        help='标注输出目录（默认: datasets/raw_det_labels')
    parser.add_argument('--model',     type=str,  default=None,
                        help='检测模型路径 (.pt/.onnx)，用于半自动预标注')
    args = parser.parse_args()

    img_dir   = args.img_dir
    if args.label_dir:
        label_dir = args.label_dir
    else:
        # images/train → labels/train
        parts = img_dir.parts
        try:
            img_idx   = [p.lower() for p in parts].index('images')
            label_dir = Path(*parts[:img_idx]) / 'labels' / Path(*parts[img_idx+1:])
        except ValueError:
            label_dir = img_dir.parent.parent / 'labels' / img_dir.name

    if not img_dir.exists():
        print(f"[错误] 图片目录不存在: {img_dir}")
        sys.exit(1)

    DetReviewTool(img_dir, label_dir, args.model)


if __name__ == '__main__':
    main()

