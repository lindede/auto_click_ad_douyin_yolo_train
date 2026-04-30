"""
settings.py —— 全局配置
所有路径、分辨率、训练超参数均在此定义。
"""

from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── scrcpy 截图分辨率 ────────────────────────────────────────
IMG_WIDTH  = 424   # 像素
IMG_HEIGHT = 944   # 像素

# ── YOLO 训练参数 ────────────────────────────────────────────
YOLO_IMGSZ  = 960   # 训练输入尺寸（标准640，会letterbox填充）
EPOCHS_CLS  = 100   # 分类模型训练轮数
EPOCHS_DET  = 100   # 检测模型训练轮数
BATCH_SIZE  = 16    # 批大小（CPU训练可适当减小）
PATIENCE    = 20    # 早停耐心值
WORKERS     = 4     # DataLoader线程数

# ── 预训练权重（首次运行会自动下载）────────────────────────────
CLS_MODEL_BASE = "yolo11n-cls.pt"
DET_MODEL_BASE = "yolo11n.pt"

# ── 数据集路径 ───────────────────────────────────────────────
DATASETS_DIR = ROOT / "datasets"
RAW_DIR      = DATASETS_DIR / "raw"        # 原始截图目录（按需子文件夹）
RAW_DET_LABELS = DATASETS_DIR / "raw_det_labels"  # 检测标注（与 raw 图片 stem 对应的 .txt）
CLS_DIR      = DATASETS_DIR / "classification"  # 分类数据集（工具自动生成）
DET_DIR      = DATASETS_DIR / "detection"       # 检测数据集

# ── 标注文件 ─────────────────────────────────────────────────
CLS_MANIFEST = DATASETS_DIR / "cls_annotations.json"   # 分类标注清单

# ── 训练输出 ─────────────────────────────────────────────────
RUNS_DIR     = ROOT / "runs"
RUNS_CLS_DIR = RUNS_DIR / "cls"
RUNS_DET_DIR = RUNS_DIR / "det"

# ── 推理置信度阈值 ───────────────────────────────────────────
CLS_CONF_THRESH = 0.6   # 页面分类置信度阈值
DET_CONF_THRESH = 0.5   # 目标检测置信度阈值

# ── 训练集/验证集划分比例 ─────────────────────────────────────
TRAIN_RATIO = 0.8   # 80% 训练，20% 验证

