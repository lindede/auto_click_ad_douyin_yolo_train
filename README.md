# 抖音广告自动点击 — YOLO 训练框架

基于 **Ultralytics YOLOv11** 的两阶段视觉识别系统，用于识别 scrcpy 截屏图片中的抖音 APP 页面类型及页面内 UI 元素，为自动点击广告任务提供视觉支撑。

---

## 运行环境

| 项目 | 版本 / 说明 |
|------|-------------|
| Python | 3.11.15 |
| PyTorch | 2.11.0 (CPU) |
| Ultralytics | 8.4.26 |
| conda 环境 | `yolo` |
| 运行设备 | CPU（Intel Core i7-7700） |
| 截图分辨率 | **255 × 567**（scrcpy 固定分辨率） |
| 推理频率 | 1 张 / 秒 |

---

## 系统设计

### 两阶段推理流程

```
scrcpy 截图 (255×567)
        │
        ▼
┌─────────────────────┐
│  第一阶段：页面分类   │  yolo11n-cls  (~1.5M 参数)
│  判断当前在哪个页面   │
└────────┬────────────┘
         │ PageClass
         ▼
┌─────────────────────┐
│  第二阶段：目标检测   │  yolo11n      (~2.6M 参数)
│  定位页面内 UI 元素   │
└─────────────────────┘
         │ List[Detection]
         ▼
    自动点击逻辑
```

选用 `yolo11n` 系列（nano）原因：参数量最小，适合在 PC CPU 上 1秒/帧的推理场景。

---

## 标签定义

> 所有标签统一在 `configs/labels.py` 中维护，新增类别只需修改该文件。

### 页面分类（8 类）— `PageClass`

| ID | 类别名 | 说明 |
|----|--------|------|
| 0 | `DouyinEarningTasksPageInfo` | 抖音 APP — 赚钱任务页面 |
| 1 | `DouyinOpenChestRewardsPageInfo` | 抖音 APP — 获得开宝箱奖励页面 |
| 2 | `DouyinTotalRewardsPageInfo` | 抖音 APP — 累计获得奖励页面 |
| 3 | `DouyinNextAdPageInfo` | 抖音 APP — 再看一个视频额外获得页面 |
| 4 | `DouyinAdShowPageInfo` | 抖音 APP — 广告展示页面 |
| 5 | `DouyinHomePageInfo` | 抖音 APP — 首页页面 |
| 6 | `AndroidDesktopPageInfo` | Android 系统 — 桌面页面 |
| 7 | `OtherPageInfo` | 其他页面 |

### 目标检测（18 类）— `DetClass`

| ID | 类别名 | 说明 |
|----|--------|------|
| 0 | `system_time` | 系统时间 |
| 1 | `earning_task_ready_button` | 赚钱任务 — 可点击按钮 |
| 2 | `earning_task_waiting_button` | 赚钱任务 — 等待中按钮 |
| 3 | `coins` | 金币图标 / 数量 |
| 4 | `watch_ad_to_earn_button` | 看广告赚金币按钮 |
| 5 | `happy_to_receive_button` | 开心收下按钮 |
| 6 | `rate_and_get_coins_button` | 评分获得金币按钮 |
| 7 | `ad_title` | 广告标题 |
| 8 | `get_reward_ready_button` | 领取奖励 — 可点击按钮 |
| 9 | `get_reward_waiting_button` | 领取奖励 — 等待中按钮 |
| 10 | `receive_reward_to_next_adA_button` | 领取奖励并看下一个广告按钮 |
| 11 | `insist_exit_button` | 坚持退出按钮 |
| 12 | `douyin_home_button` | 抖音首页按钮 |
| 13 | `douyin_me_button` | 抖音我的按钮 |
| 14 | `douyin_lite_lanch_icon` | 抖音启动图标 |
| 15 | `kuaishou_lite_lanch_icon` | 快手启动图标 |
| 16 | `open_box_for_coins_button` | 开宝箱得金币按钮 |
| 17 | `open_box_for_coins_waiting_button` | 开宝箱得金币等待按钮 |

---

## 目录结构

```
auto_click_ad_douyin_yolo_train/
├── configs/
│   ├── labels.py            # ⭐ 唯一标签数据源（PageClass / DetClass）
│   ├── settings.py          # 全局配置（路径 / 分辨率 / 超参数）
│   ├── cls_dataset.yaml     # 分类数据集配置（自动生成）
│   └── det_dataset.yaml     # 检测数据集配置（自动生成）
├── datasets/
│   ├── raw/                 # 原始 scrcpy 截图（手动放入）
│   ├── cls_annotations.json # 分类标注清单（cls_annotator 生成）
│   ├── classification/      # 分类数据集（dataset_builder 生成）
│   │   ├── train/<ClassName>/
│   │   └── val/<ClassName>/
│   └── detection/           # 检测数据集（手动 + 工具辅助）
│       ├── images/train|val/
│       └── labels/train|val/
├── tools/
│   ├── cls_annotator.py     # 🖼️ 快速分类标注 GUI（键盘驱动）
│   ├── dataset_builder.py   # 📦 数据集构建 & yaml 生成
│   └── det_review_tool.py   # 🎯 半自动检测标注审核工具
├── train/
│   ├── train_cls.py         # 训练分类模型
│   └── train_det.py         # 训练检测模型
├── export/
│   └── export_onnx.py       # 导出 ONNX（CPU 推理加速）
├── inference/
│   └── predictor.py         # 两阶段推理封装
├── runs/                    # 训练输出（自动生成）
│   ├── cls/<name>/
│   └── det/<name>/
├── yolo11n-cls.pt           # 预训练权重（首次训练时自动下载）
├── yolo11n.pt
└── requirements.txt
```

---

## 安装

```bash
conda activate yolo
pip install -r requirements.txt
```

---

## 完整工作流程

### 第一阶段：页面分类模型

#### 1. 标注截图页面类型

将 scrcpy 截图放入 `datasets/raw/`，运行分类标注工具：

```bash
python tools/cls_annotator.py --src datasets/raw
```

**操作说明：**

| 按键 | 动作 |
|------|------|
| `1` ~ `8` | 给当前图片打对应页面标签，自动跳下一张 |
| `←` / `→` | 上一张 / 下一张（不标注） |
| `u` | 撤销当前图片标注 |
| `q` / `Esc` | 保存并退出 |

标注结果保存至 `datasets/cls_annotations.json`。

#### 2. 构建分类数据集

```bash
python tools/dataset_builder.py --task cls
```

自动按 80/20 比例划分 train/val，生成 `configs/cls_dataset.yaml`。

#### 3. 训练分类模型

```bash
python train/train_cls.py
# 自定义参数示例
python train/train_cls.py --epochs 150 --batch 8 --name cls_v2
# 从中断处继续
python train/train_cls.py --resume
```

训练结果保存在 `runs/cls/<name>/`，详见 [训练输出说明](#训练输出说明)。

---

### 第二阶段：目标检测模型

#### 冷启动（无初始模型）— 纯手工标注

```bash
# 初始化检测数据集目录结构和 yaml
python tools/dataset_builder.py --task det

# 打开标注工具，手工画框（建议每类至少 30 张）
python tools/det_review_tool.py
```

**检测标注工具操作说明：**

| 操作 | 动作 |
|------|------|
| 左键拖拽空白区域 | 画新检测框（默认 class 0，需按数字键改类别） |
| 左键单击已有框 | 选中框（高亮显示） |
| 右键单击框 | 删除该框 |
| `1` ~ `9` | 修改选中框的类别 |
| `Delete` / `d` | 删除选中框 |
| `a` | 调用模型自动预标注（需指定 --model） |
| `c` | 清空当前图片所有框 |
| `s` | 保存当前图片标注 |
| `n` / `→` | 保存并跳到下一张 |
| `p` / `←` | 保存并跳到上一张 |
| `q` / `Esc` | 保存并退出 |

#### 训练初始检测模型

```bash
python train/train_det.py
```

#### 半自动标注（有初始模型后）

```bash
python tools/det_review_tool.py --model runs/det/det_v1/weights/best.pt
# 按 a 键自动预标注 → 人工审核修改 → n 保存下一张
```

重复「补充标注 → 训练」循环，直到检测精度满足需求。

---

### 第三阶段：导出 ONNX

```bash
# 导出分类模型
python export/export_onnx.py --weights runs/cls/cls_v1/weights/best.pt --task cls

# 导出检测模型
python export/export_onnx.py --weights runs/det/det_v1/weights/best.pt --task det
```

ONNX 格式无需 PyTorch 运行时，CPU 推理速度更快。

---

### 调用推理模块

```python
from inference.predictor import Predictor

p = Predictor(
    cls_model='runs/cls/cls_v1/weights/best.onnx',
    det_model='runs/det/det_v1/weights/best.onnx',
)

result = p.predict_file('datasets/raw/frame_001.png')
print(result.page_class)       # PageClass.DouyinAdShowPageInfo
print(result.page_confidence)  # 0.95
print(result.detections)       # [Detection(watch_ad_to_earn_button, ...), ...]
```

---

## 训练输出说明

每次训练在 `runs/cls/<name>/` 或 `runs/det/<name>/` 下生成：

| 文件 | 说明 |
|------|------|
| `weights/best.pt` | ⭐ 验证集最优权重，用于导出和推理 |
| `weights/last.pt` | 最后一个 epoch 权重，用于 `--resume` 继续训练 |
| `args.yaml` | 本次训练完整参数快照，便于复现 |
| `results.csv` | 每个 epoch 的 loss / accuracy / lr 数据 |
| `results.png` | 训练曲线可视化图 |
| `confusion_matrix.png` | 混淆矩阵（原始数量） |
| `confusion_matrix_normalized.png` | 混淆矩阵（归一化百分比） |
| `train_batch*.jpg` | 训练样本预览（含数据增强效果） |
| `val_batch*_labels.jpg` | 验证集真实标签可视化 |
| `val_batch*_pred.jpg` | 验证集预测结果可视化 |

---

## 配置说明

### `configs/settings.py` 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IMG_WIDTH / IMG_HEIGHT` | `255 / 567` | scrcpy 截图固定分辨率 |
| `YOLO_IMGSZ` | `640` | YOLO 训练输入尺寸（letterbox 填充） |
| `EPOCHS_CLS / EPOCHS_DET` | `100` | 最大训练轮数（early stop 由 patience 控制） |
| `BATCH_SIZE` | `16` | CPU 训练可调小至 8 |
| `PATIENCE` | `20` | 连续 N 轮无提升则提前停止 |
| `TRAIN_RATIO` | `0.8` | 训练集比例（0.2 为验证集） |
| `CLS_CONF_THRESH` | `0.6` | 页面分类置信度阈值 |
| `DET_CONF_THRESH` | `0.5` | 目标检测置信度阈值 |

### 新增标签

1. 打开 `configs/labels.py`
2. 在 `PageClass` 或 `DetClass` 枚举末尾追加新类别
3. 重新运行 `dataset_builder.py` 生成新的 yaml
4. 补充该类别的标注数据后重新训练

---

## 常见问题

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Classification datasets must be a directory` | 分类训练 `data` 参数不能传 yaml | 传 `datasets/classification/` 目录路径（已修复） |
| `det_dataset.yaml does not exist` | 未初始化检测数据集 | 先运行 `python tools/dataset_builder.py --task det` |
| PowerShell 命令报错 | `&&` 在 PowerShell 中无效 | 使用 `;` 分隔命令，或分两行执行 |

