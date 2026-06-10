# 认知科学导论大作业 —— EEG 跨会话二分类

基于脑电（EEG）的单试次二分类任务：区分 `background`（背景刺激）与 `target`（目标刺激）两类诱发响应。
训练集为 `sess1`+`sess2`，测试集为被完整留出的 `sess3`（属于跨会话泛化场景）。

---

## 一、环境依赖

- Python 3.10+
- PyTorch、NumPy、Pandas

```bash
pip install torch numpy pandas
```

有 GPU 会自动使用 GPU，否则用 CPU。

## 二、数据准备

将教师提供的数据整理为以下结构，放在项目根目录下：

```text
data/
├── train/            # 训练集 EEG .npy 文件
├── test/             # 测试集 EEG .npy 文件
└── train_labels.csv  # 训练标签（至少含 eeg_file, label 两列）
```

每个 `.npy` 样本形状为 `(1, 59, 282)`：1 个输入通道、59 个电极通道、282 个时间点。

## 三、使用方法

```bash
# 1.（可选）训练单个模型，并查看诚实的跨会话验证准确率
python train.py

# 2. 训练最终提交用的集成模型（8 个随机种子，全量数据）
python train_final.py

# 3.（可选）端到端估计真实跨会话准确率，并校准决策阈值
python validate_final.py

# 4. 对测试集生成预测结果
python test.py
```

运行 `test.py` 后，会在 `res/predictions.csv` 生成两列：`eeg_file, prediction`（即提交结果）。

> 若只想直接得到预测结果：仓库已附带训练好的模型（`models/`），按上面放好数据后直接运行第 4 步 `python test.py` 即可。

## 四、文件说明

| 文件 / 目录 | 作用 |
| --- | --- |
| `model.py` | 模型定义（改进版 EEGNet 轻量卷积网络，约 2562 参数） |
| `load_data.py` | 数据读取、标准化、数据增强、数据划分 |
| `train.py` | 单模型训练 + 按会话留一的跨会话验证 |
| `train_final.py` | 多种子全量训练，产出最终提交的集成模型 |
| `validate_final.py` | 端到端诚实跨会话验证 + 阈值校准 |
| `tune_threshold.py` | 决策阈值调优 |
| `test.py` | 集成 + 时移 TTA 推理，生成 `res/predictions.csv` |
| `experiments.py` | 可配置实验台（被上述脚本复用） |
| `utils.py` | 随机种子固定、设备选择 |
| `best_config.json` | 最终选定的超参数配置 |
| `models/` | 训练好的模型权重 + `ensemble_meta.json` 元信息 |
| `res/` | 测试集预测结果 |
| `data/` | 数据放置位置 |

## 五、方法简述

- 采用 EEGNet 风格的轻量 CNN，参数极少，适合小样本 EEG 数据、不易过拟合。
- 用「按会话留一（LSO）」做诚实的跨会话验证，避免同会话信息泄露造成的虚高。
- 最终用 8 个随机种子在全量数据上训练做集成，推理时叠加时移 TTA，提升稳健性。
