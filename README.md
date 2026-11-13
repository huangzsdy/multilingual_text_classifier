# Multilingual Text Classifier

[English](#english) | [中文](#中文)

---

## 📖 简介

本项目实现基于预训练语言模型的多语言文本二分类（正/负样本分类）。支持15+种语言的文本分类任务，包含模型训练、批量推理、断点续传等功能。

模型路径通过配置文件统一管理，支持任意 Hugging Face 预训练模型（如 XLM-RoBERTa、BERT、RoBERTa 等）。

## 🛠️ 环境要求

```bash
pip install -r requirements.txt
```

### 核心依赖

- `transformers>=4.30.0` - Hugging Face Transformer 模型库
- `torch>=2.0.0` - PyTorch 深度学习框架
- `scikit-learn>=1.2.0` - 机器学习工具
- `tqdm>=4.65.0` - 进度条
- `tensorboard>=2.12.0` - 训练可视化

## 📁 项目结构

```
multilingual_text_classifier/
├── scripts/
│   ├── train.py                   # 模型训练脚本
│   ├── inference.py               # 基础推理脚本
│   ├── inference_stream_v6.py     # 流式推理（续写、目录输出、多GPU）
│   ├── inference_stream_v5.py     # 流式推理脚本v5
│   ├── inference_stream_v3.py     # 流式推理脚本v3
│   ├── filter_pos_jsonl.py       # 按概率过滤正样本
│   ├── jsonl_classify.py          # JSONL文件分类
│   ├── filter_testset.py          # 测试集过滤
│   └── classify.sh                # 分类脚本
├── tools/
│   ├── utils.py                   # 工具函数
│   └── calculate_metric_v2.py     # 指标计算
├── configs/
│   ├── config.json                # 主配置文件（模型路径在此配置）
│   ├── config_test.json           # 测试配置
│   └── config_merge_train.json    # 训练配置
├── data/
│   ├── positive_samples.jsonl     # 正样本数据
│   └── negative_samples.jsonl     # 负样本数据
├── train.sh                       # 训练入口脚本
├── process_all_lang.sh            # 多语言处理脚本
└── run_inference_folder.sh        # 批量推理脚本
```

## ⚙️ 配置说明

模型路径在 `configs/config.json` 中配置：

```json
{
  "model_config": {
    "model_name": "FacebookAI/xlm-roberta-base",
    "num_labels": 2,
    "max_length": 512
  },
  "training_config": {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

支持的主流模型：
- `FacebookAI/xlm-roberta-base` / `xlm-roberta-large`
- `bert-base-multilingual-cased`
- `roberta-base` / `roberta-large`
- 任意 Hugging Face 支持的文本分类模型

## 🚀 快速开始

### 1. 模型训练

```bash
# 使用默认配置训练
bash train.sh

# 或指定配置文件
python scripts/train.py --config configs/config.json
```

### 2. 批量推理

```bash
# 按 run_inference_folder.sh 中的 TASKS 配置批量处理目录
bash run_inference_folder.sh

# v6：多个输入合并并续写到一个 JSONL
python scripts/inference_stream_v6.py \
    --model_path ./results/checkpoint-1000 \
    --jsonl_files data/a.jsonl data/b.jsonl \
    --output_file output/predictions.jsonl \
    --checkpoint_dir checkpoints/job1 \
    --batch_size 32 \
    --use_amp

# v6：每个输入文件分别续写到输出目录中的同名文件
python scripts/inference_stream_v6.py \
    --model_path ./results/checkpoint-1000 \
    --jsonl_files data/a.jsonl data/b.jsonl \
    --output_dir output/predictions \
    --checkpoint_dir checkpoints/job2 \
    --batch_size 32 \
    --use_amp
```

`--output_file` 与 `--output_dir` 必须且只能指定一个。v6 不会清空已有输出；重新运行时会读取 `checkpoint_dir` 中按输入文件保存的文本哈希，只推理尚未处理的文本，并将新结果追加到现有输出。请为不同任务使用独立的 checkpoint 目录，并保留输出文件和 checkpoint 的对应关系。

### `run_inference_folder.sh`

该脚本适合按目录批量启动基础推理任务：

- 在脚本的 `TASKS` 中配置输入目录、模型路径、输出目录和日志文件。
- 只扫描输入目录第一层的 `*.jsonl`；每个输入文件写入输出目录中的同名文件。
- 可通过 `GPUS`、`PER_GPU` 和 `BATCH_SIZE` 环境变量调整 GPU 列表、每卡任务上限和推理批大小。
- 每个 JSONL 都会单独启动 `scripts/inference.py` 并加载模型。
- `scripts/inference.py` 会覆盖已存在的同名输出文件，不提供 checkpoint 续传。需要流式处理和断点续写时应使用 `inference_stream_v6.py`。

示例：

```bash
GPUS=0,1 PER_GPU=1 BATCH_SIZE=64 bash run_inference_folder.sh
```

### 3. 过滤正样本

```bash
# 单文件过滤
python scripts/filter_pos_jsonl.py input.jsonl output.jsonl 0.5

# 目录批量过滤
python scripts/filter_pos_jsonl.py input_dir/ output_dir/ 0.5
```

## ⚡ 核心功能

### 训练 (scripts/train.py)

- 支持任意 Hugging Face 预训练模型
- 多语言联合训练
- Early Stopping 防止过拟合
- TensorBoard 可视化
- Checkpoint 保存与恢复

### 推理 (`scripts/inference_stream_v6.py`)

- **流式处理**：大文件按读取批次处理，避免一次性载入全部数据
- **断点续写**：基于文本 MD5 哈希记录 checkpoint，不清空已有输出，只追加新结果
- **两种输出模式**：`--output_file` 合并输出，或 `--output_dir` 按输入文件分别输出
- **多 GPU 并行**：保留 DataParallel 多卡加速
- **混合精度**：支持 AMP/FP16 推理
- **自动跳过**：重新运行时跳过 checkpoint 已记录的文本

### 断点续传机制

```
首次运行:
  输入文件 → 处理 → 输出文件
              ↓
         checkpoint.txt (记录已处理文本的MD5哈希)

再次运行:
  读取checkpoint → 跳过已处理文本 → 只处理新文本
```

## 📊 支持的语言

| 语言 | 代码 |
|------|------|
| 中文 | zh |
| 英文 | en |
| 日语 | ja |
| 阿拉伯语 | ar |
| 德语 | de |
| 法语 | fr |
| 西班牙语 | es |
| 俄语 | ru |
| 韩语 | ko |
| 葡萄牙语 | pt |
| 意大利语 | it |
| 印地语 | hi |
| 希伯来语 | he |
| 波兰语 | pl |
| 荷兰语 | nl |

## 📝 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
bash train.sh

# 多语言处理
bash process_all_lang.sh

# 按目录批量运行基础推理任务（会覆盖同名输出）
bash run_inference_folder.sh

# 计算指标
bash calculate_metrics.sh
```

## 📄 许可证

MIT License

---

# English

## Multilingual Text Classifier

## 📖 Introduction

This project implements multilingual text binary classification (positive/negative) using pre-trained language models. Supports 15+ languages with model training, batch inference, and checkpoint resume capabilities.

Model paths are managed through configuration files, supporting any Hugging Face pre-trained model (XLM-RoBERTa, BERT, RoBERTa, etc.).

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `transformers>=4.30.0` - Hugging Face Transformer
- `torch>=2.0.0` - PyTorch
- `scikit-learn>=1.2.0` - Machine Learning
- `tqdm>=4.65.0` - Progress bar
- `tensorboard>=2.12.0` - Training visualization

## 📁 Project Structure

```
multilingual_text_classifier/
├── scripts/
│   ├── train.py                   # Model training
│   ├── inference.py               # Basic inference
│   ├── inference_stream_v6.py     # Streaming resume, directory output, multi-GPU
│   ├── inference_stream_v5.py     # Streaming inference v5
│   ├── inference_stream_v3.py     # Streaming inference v3
│   ├── filter_pos_jsonl.py       # Filter positive samples by probability
│   ├── jsonl_classify.py          # JSONL classification
│   └── classify.sh                # Classification script
├── tools/
│   ├── utils.py                   # Utilities
│   └── calculate_metric_v2.py     # Metrics calculation
├── configs/
│   ├── config.json                # Main config (model path configured here)
│   ├── config_test.json           # Test config
│   └── config_merge_train.json    # Training config
└── ...
```

## ⚙️ Configuration

Model path is configured in `configs/config.json`:

```json
{
  "model_config": {
    "model_name": "FacebookAI/xlm-roberta-base",
    "num_labels": 2,
    "max_length": 512
  },
  "training_config": {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

Supported models:
- `FacebookAI/xlm-roberta-base` / `xlm-roberta-large`
- `bert-base-multilingual-cased`
- `roberta-base` / `roberta-large`
- Any Hugging Face text classification model

## 🚀 Quick Start

### Training

```bash
bash train.sh
```

### Batch Inference

```bash
# Process folders configured in TASKS
bash run_inference_folder.sh

# v6: merge multiple inputs into one append-only JSONL
python scripts/inference_stream_v6.py \
    --model_path ./results/checkpoint-1000 \
    --jsonl_files data/a.jsonl data/b.jsonl \
    --output_file output/predictions.jsonl \
    --checkpoint_dir checkpoints/job1 \
    --batch_size 32 \
    --use_amp

# v6: write each input to a same-named file in an output directory
python scripts/inference_stream_v6.py \
    --model_path ./results/checkpoint-1000 \
    --jsonl_files data/a.jsonl data/b.jsonl \
    --output_dir output/predictions \
    --checkpoint_dir checkpoints/job2 \
    --batch_size 32 \
    --use_amp
```

Exactly one of `--output_file` and `--output_dir` is required. v6 never truncates existing output. On restart, it loads per-input text hashes from `checkpoint_dir`, skips processed texts, and appends only new results. Use a dedicated checkpoint directory for each job and keep its output files together with it.

### `run_inference_folder.sh`

This script launches basic inference jobs for configured folders:

- Configure the input folder, model path, output folder, and log file in `TASKS`.
- It scans only first-level `*.jsonl` files and writes same-named output files.
- `GPUS`, `PER_GPU`, and `BATCH_SIZE` control the GPU list, intended per-GPU concurrency, and inference batch size.
- Each JSONL starts a separate `scripts/inference.py` process and reloads the model.
- `scripts/inference.py` overwrites an existing same-named output and has no inference checkpoint support. Use `inference_stream_v6.py` for streaming resume.

```bash
GPUS=0,1 PER_GPU=1 BATCH_SIZE=64 bash run_inference_folder.sh
```

### Filter Positive Samples

```bash
# Single file
python scripts/filter_pos_jsonl.py input.jsonl output.jsonl 0.5

# Directory batch processing
python scripts/filter_pos_jsonl.py input_dir/ output_dir/ 0.5
```

## ⚡ Key Features

### Training (scripts/train.py)

- Support any Hugging Face pre-trained model
- Multilingual joint training
- Early Stopping
- TensorBoard visualization
- Checkpoint saving and resuming

### Inference (`scripts/inference_stream_v6.py`)

- **Streaming**: Process large files in read batches instead of loading everything
- **Checkpoint resume**: Track texts by MD5, preserve output, and append new results
- **Output modes**: Merge with `--output_file` or split by input with `--output_dir`
- **Multi-GPU**: Retains DataParallel support
- **AMP**: Automatic Mixed Precision (FP16) inference
- **Auto-skip**: Skip texts already recorded in the checkpoint on restart

## 📊 Supported Languages

| Language | Code |
|----------|------|
| Chinese | zh |
| English | en |
| Japanese | ja |
| Arabic | ar |
| German | de |
| French | fr |
| Spanish | es |
| Russian | ru |
| Korean | ko |
| Portuguese | pt |
| Italian | it |
| Hindi | hi |
| Hebrew | he |
| Polish | pl |
| Dutch | nl |

## 📝 Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
bash train.sh

# Process all languages
bash process_all_lang.sh

# Run configured folder jobs (overwrites same-named output files)
bash run_inference_folder.sh

# Calculate metrics
bash calculate_metrics.sh
```

## 📄 License

MIT License
