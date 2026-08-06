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
│   ├── inference_stream_v5.py     # 流式推理脚本（支持断点续传）
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
# 使用 run_inference_folder.sh 进行批量推理（支持跳过已处理文件）
bash run_inference_folder.sh

# 或直接使用推理脚本
python scripts/inference_stream_v5.py \
    --model_path ./results/checkpoint-1000 \
    --jsonl_files data/test.jsonl \
    --output_file output/predictions.jsonl \
    --batch_size 32 \
    --use_amp
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

### 推理 (scripts/inference_stream_v5.py)

- **流式处理**: 大文件分批读取，避免内存溢出
- **断点续传**: 基于文本哈希的 checkpoint，记录已处理文本
- **多GPU并行**: 支持 DataParallel 多卡加速
- **混合精度**: 使用 AMP 加速推理 (FP16)
- **自动跳过**: 已处理的文本会自动跳过

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

# 批量推理（跳过已处理文件）
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
│   ├── inference_stream_v5.py     # Streaming inference (checkpoint resume)
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
# With auto-skip of processed files
bash run_inference_folder.sh

# Or use inference script directly
python scripts/inference_stream_v5.py \
    --model_path ./results/checkpoint-1000 \
    --jsonl_files data/test.jsonl \
    --output_file output/predictions.jsonl \
    --batch_size 32 \
    --use_amp
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

### Inference (scripts/inference_stream_v5.py)

- **Streaming**: Process large files in batches to avoid OOM
- **Checkpoint Resume**: Track processed texts via MD5 hash
- **Multi-GPU**: DataParallel support
- **AMP**: Automatic Mixed Precision (FP16) for faster inference
- **Auto-skip**: Automatically skips processed texts

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

# Batch inference (skip processed files)
bash run_inference_folder.sh

# Calculate metrics
bash calculate_metrics.sh
```

## 📄 License

MIT License
