#!/usr/bin/env python3
"""
推理脚本 - 使用训练好的模型进行预测
支持单文本、文本文件、JSONL文件批量预测
保证处理所有输入数据，不丢弃任何样本
支持流式读取数据，避免一次性加载大文件到内存
支持断点续传功能
"""

import os
import json
import torch
import argparse
import logging
import gc
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union, Iterator, Tuple
from tqdm import tqdm
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextClassifier:
    """文本分类器 - 保证处理所有数据，支持流式读取"""

    def __init__(self, model_path: str, device=None, max_memory_mb: int = 1024):
        """
        初始化分类器

        Args:
            model_path: 模型路径
            device: 设备 (cpu/cuda)，如果为None则自动选择
            max_memory_mb: 最大内存占用限制（MB）
        """
        self.model_path = model_path
        self.max_memory_mb = max_memory_mb

        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"使用设备: {self.device}")
        logger.info(f"内存限制: {max_memory_mb} MB")

        # 加载模型和分词器
        self.load_model()

    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # 对于CPU，使用psutil获取更准确的内存信息
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                logger.warning("未安装psutil，使用简化内存估算")
                # 简化估算：假设基础内存使用
                return 0

    def _check_memory_limit(self, additional_mb: float = 0) -> bool:
        """检查是否超过内存限制"""
        current_usage = self._get_memory_usage()
        if current_usage + additional_mb > self.max_memory_mb * 0.9:  # 留10%余量
            logger.warning(
                f"内存使用量 {current_usage:.1f}MB + {additional_mb:.1f}MB 超过限制 {self.max_memory_mb}MB 的90%")
            return False
        return True

    def _cleanup_memory(self, aggressive: bool = False):
        """清理内存

        Args:
            aggressive: 是否进行激进清理
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        if aggressive:
            # 激进清理
            for _ in range(3):
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                time.sleep(0.1)  # 给系统时间释放内存
        else:
            gc.collect()

    def _estimate_batch_memory(self, batch_size: int, max_length: int, texts: List[str] = None) -> float:
        """更精确地估算批次内存需求"""
        # 基础估算：每个token约4字节，考虑序列长度和批次大小
        base_estimate = batch_size * max_length * 4 / 1024 / 1024  # MB

        # 如果提供了实际文本，进行更精确的估算
        if texts and len(texts) > 0:
            avg_length = sum(len(text) for text in texts) / len(texts)
            actual_estimate = batch_size * min(avg_length, max_length) * 4 / 1024 / 1024
            # 取较大值，保守估计
            return max(base_estimate, actual_estimate) * 1.5  # 增加50%安全边际

        return base_estimate * 2.0  # 保守估计

    def load_model(self):
        """加载模型和分词器"""
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式

            logger.info(f"模型加载成功: {self.model_path}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def predict_with_guaranteed_completion(self, texts: Union[str, List[str]], output_file: str,
                                           label_mapping=None, max_length=512, initial_batch_size=32,
                                           processed_file_lasttime: str = None):
        """
        保证处理所有文本的预测方法，支持断点续传

        Args:
            texts: 文本列表
            output_file: 输出文件路径
            label_mapping: 标签映射
            max_length: 最大序列长度
            initial_batch_size: 初始批次大小
            processed_file_lasttime: 上次处理中断的结果文件，用于跳过已处理部分

        Returns:
            成功处理的样本数量（应该等于len(texts)）
        """
        if label_mapping is None:
            label_mapping = {0: "negative", 1: "positive"}

        if isinstance(texts, str):
            texts = [texts]

        total_samples = len(texts)
        if total_samples == 0:
            logger.warning("没有文本需要处理")
            return 0

        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否存在上次处理的结果文件
        processed_hashes = set()
        start_index = 0

        if processed_file_lasttime and os.path.exists(processed_file_lasttime):
            logger.info(f"检测到上次处理结果文件: {processed_file_lasttime}")
            logger.info("正在分析已处理样本，准备跳过...")

            try:
                # 读取上次处理的结果，提取已处理的文本哈希
                with open(processed_file_lasttime, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                # 计算文本哈希作为唯一标识
                                text_hash = hash(data['text'])
                                processed_hashes.add(text_hash)
                        except json.JSONDecodeError:
                            continue

                logger.info(f"发现 {len(processed_hashes)} 个已处理样本")

                # 找到第一个未处理的样本索引
                for i, text in enumerate(texts):
                    if hash(text) not in processed_hashes:
                        start_index = i
                        break
                else:
                    # 所有样本都已处理
                    start_index = total_samples

                logger.info(f"将从第 {start_index} 个样本开始处理 (跳过 {start_index} 个已处理样本)")

            except Exception as e:
                logger.error(f"读取上次处理结果文件失败: {e}，将从头开始处理")
                start_index = 0
        else:
            # 清空或创建输出文件（如果是全新开始）
            if not processed_file_lasttime or not os.path.exists(processed_file_lasttime):
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("")  # 清空文件

        # 初始内存状态
        initial_memory = self._get_memory_usage()
        logger.info(f"开始处理 {total_samples - start_index} 个未处理样本，初始内存: {initial_memory:.1f}MB")

        processed_count = 0
        current_index = start_index
        current_batch_size = initial_batch_size
        retry_count = 0
        max_retries = 10  # 最大重试次数

        # 使用进度条（只显示未处理的部分）
        pbar = tqdm(total=total_samples - start_index, desc="处理进度")

        # 处理所有文本
        while current_index < total_samples:
            # 获取当前批次，跳过已处理的文本
            batch_texts = []
            batch_indices = []

            for i in range(current_index, min(current_index + current_batch_size, total_samples)):
                text = texts[i]
                if hash(text) not in processed_hashes:  # 只处理未处理的文本
                    batch_texts.append(text)
                    batch_indices.append(i)

            actual_batch_size = len(batch_texts)

            if actual_batch_size == 0:
                # 当前批次都是已处理的文本，直接跳过
                current_index += current_batch_size
                continue

            # 估算内存需求
            estimated_memory = self._estimate_batch_memory(actual_batch_size, max_length, batch_texts)

            # 检查内存限制
            memory_ok = self._check_memory_limit(estimated_memory)

            if not memory_ok:
                if current_batch_size > 1:
                    # 减小批次大小
                    new_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"内存紧张，减小批次大小: {current_batch_size} -> {new_batch_size}")
                    current_batch_size = new_batch_size

                    # 激进清理内存
                    self._cleanup_memory(aggressive=True)
                    retry_count += 1

                    if retry_count > max_retries:
                        logger.error(f"连续 {max_retries} 次重试后仍内存紧张，尝试单样本处理")
                        current_batch_size = 1
                        retry_count = 0

                    continue  # 重新尝试当前批次
                else:
                    # 批次大小已经是1，但仍然内存不足
                    logger.warning("批次大小为1仍内存不足，尝试激进清理并等待")
                    self._cleanup_memory(aggressive=True)
                    time.sleep(1)  # 等待系统释放内存

                    # 再次检查
                    if not self._check_memory_limit(estimated_memory):
                        logger.error("系统内存严重不足，无法继续处理")
                        raise MemoryError("系统内存不足，无法处理即使单个样本")

            try:
                # 重置重试计数
                retry_count = 0

                # 分词
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)

                # 转换为CPU numpy数组
                batch_predictions = predictions.cpu().numpy().tolist()
                batch_probabilities = probabilities.cpu().numpy().tolist()

                # 写入结果到文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    for j, (text, pred, probs, orig_idx) in enumerate(
                            zip(batch_texts, batch_predictions, batch_probabilities, batch_indices)):
                        result = {
                            "text": text,
                            "label": int(pred),
                            "label_name": label_mapping.get(int(pred), f"label_{pred}"),
                            "confidence": float(max(probs)),
                            "probabilities": {label_mapping.get(k, f"label_{k}"): float(v)
                                              for k, v in enumerate(probs)},
                            "batch_index": current_index // current_batch_size,
                            "sample_index": orig_idx  # 使用原始索引
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                # 更新进度
                processed_count += actual_batch_size
                current_index += current_batch_size
                pbar.update(actual_batch_size)

                # 标记这些文本为已处理
                for text in batch_texts:
                    processed_hashes.add(hash(text))

                # 尝试增加批次大小（如果内存允许）
                if current_batch_size < initial_batch_size:
                    # 检查是否可以增加批次大小
                    test_batch_size = min(current_batch_size * 2, initial_batch_size)
                    test_estimate = self._estimate_batch_memory(test_batch_size, max_length)

                    if self._check_memory_limit(test_estimate):
                        logger.info(f"内存充足，增加批次大小: {current_batch_size} -> {test_batch_size}")
                        current_batch_size = test_batch_size

                # 清理内存
                del inputs, outputs, logits, probabilities, predictions
                self._cleanup_memory()

                # 定期报告状态
                if processed_count % 100 == 0:
                    current_memory = self._get_memory_usage()
                    logger.info(
                        f"已处理 {processed_count}/{total_samples - start_index} 未处理样本, 内存使用: {current_memory:.1f}MB")

            except Exception as e:
                logger.error(f"处理批次失败 (索引 {current_index}-{current_index + actual_batch_size - 1}): {e}")

                # 尝试减小批次大小重试
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"处理失败，减小批次大小重试: {current_batch_size}")
                else:
                    # 批次大小已经是1，跳过当前样本（这是最后的手段）
                    logger.error(f"无法处理样本 {current_index}，跳过: {texts[current_index][:100]}...")
                    # 记录错误但继续处理下一个
                    with open(output_file, 'a', encoding='utf-8') as f:
                        error_result = {
                            "text": texts[current_index],
                            "label": -1,
                            "label_name": "ERROR",
                            "confidence": 0.0,
                            "probabilities": {"ERROR": 1.0},
                            "error": str(e),
                            "sample_index": current_index
                        }
                        f.write(json.dumps(error_result, ensure_ascii=False) + '\n')

                    # 标记为已处理（即使是错误状态）
                    processed_hashes.add(hash(texts[current_index]))
                    processed_count += 1
                    current_index += 1
                    pbar.update(1)

        pbar.close()

        final_memory = self._get_memory_usage()
        success_rate = (processed_count / (total_samples - start_index)) * 100 if (
                                                                                              total_samples - start_index) > 0 else 100

        if processed_count == (total_samples - start_index):
            logger.info(
                f"✅ 处理完成: 新增 {processed_count} 个样本，跳过 {start_index} 个已处理样本 (100%)，最终内存: {final_memory:.1f}MB")
        else:
            logger.warning(
                f"⚠️ 处理部分完成: 新增 {processed_count}/{(total_samples - start_index)} 未处理样本 ({success_rate:.1f}%)")

        return processed_count

    # 保留原有方法兼容性
    def predict_with_disk_cache(self, texts: Union[str, List[str]], output_file: str,
                                label_mapping=None, max_length=512, batch_size=32,
                                processed_file_lasttime: str = None):
        """兼容旧接口，使用保证完成的新方法"""
        return self.predict_with_guaranteed_completion(
            texts, output_file, label_mapping, max_length, batch_size, processed_file_lasttime)

    def predict(self, texts: Union[str, List[str]], max_length=512, batch_size=32, return_probs=False):
        """
        对文本进行分类预测（内存中处理，不保证大数据集）
        """
        if isinstance(texts, str):
            texts = [texts]

        # 对于小批量数据，使用原有逻辑
        if len(texts) <= 1000:  # 小数据集直接处理
            return self._predict_small_batch(texts, max_length, batch_size, return_probs)
        else:
            # 大数据集使用保证完成的方法
            temp_file = "temp_predictions.jsonl"
            self.predict_with_guaranteed_completion(texts, temp_file, max_length=max_length,
                                                    initial_batch_size=batch_size)

            # 读取结果
            with open(temp_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]

            # 清理临时文件
            os.remove(temp_file)

            if return_probs:
                predictions = [r['label'] for r in results]
                probabilities = [list(r['probabilities'].values()) for r in results]
                return predictions, probabilities
            else:
                return [r['label'] for r in results]

    def _predict_small_batch(self, texts: List[str], max_length=512, batch_size=32, return_probs=False):
        """处理小批量数据的原有逻辑"""
        all_predictions = []
        all_probabilities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

        if return_probs:
            return all_predictions, all_probabilities
        return all_predictions


def load_jsonl_file_streaming(file_path: str, text_field: str = "content",
                              max_samples: int = None, batch_size: int = 1000) -> Iterator[Tuple[List[dict], int]]:
    """
    流式读取JSONL文件，按批次返回数据，避免一次性加载大文件到内存

    Args:
        file_path: JSONL文件路径
        text_field: 包含文本的字段名
        max_samples: 最大读取样本数
        batch_size: 每个批次的大小

    Yields:
        (batch_data, total_count): 批次数据和总计数
    """
    batch_data = []
    total_count = 0
    current_batch = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_samples and total_count >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())
                    if text_field in data and data[text_field].strip():
                        batch_data.append(data)
                        total_count += 1

                        # 当批次达到指定大小时返回
                        if len(batch_data) >= batch_size:
                            yield batch_data, total_count
                            batch_data = []
                            current_batch += 1

                except json.JSONDecodeError:
                    logger.warning(f"第 {line_num + 1} 行JSON解析错误，已跳过")
                    continue

        # 返回最后一个不完整的批次
        if batch_data:
            yield batch_data, total_count

    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        raise


def load_jsonl_file(file_path: str, text_field: str = "content", max_samples: int = None):
    """从JSONL文件读取数据（兼容旧版本，一次性读取）"""
    texts = []
    original_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in tqdm(f, desc=f"读取 {os.path.basename(file_path)}"):
                if max_samples and line_count >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())
                    if text_field in data and data[text_field].strip():
                        texts.append(data[text_field].strip())
                        original_data.append(data)
                        line_count += 1
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        raise

    logger.info(f"从 {file_path} 读取了 {len(texts)} 个有效样本")
    return texts, original_data


def guarantee_process_all_files_streaming(classifier, jsonl_files: List[str], output_file: str,
                                          text_field: str = "content", max_length: int = 512,
                                          batch_size: int = 32, max_samples: int = None,
                                          read_batch_size: int = 1000,
                                          processed_file_lasttime: str = None):
    """
    流式处理所有文件，保证处理所有数据，避免一次性加载大文件到内存
    支持断点续传功能

    Args:
        classifier: 文本分类器实例
        jsonl_files: JSONL文件路径列表
        output_file: 输出文件路径
        text_field: 包含文本的字段名
        max_length: 最大序列长度
        batch_size: 预测批次大小
        max_samples: 最大样本数
        read_batch_size: 读取文件的批次大小
        processed_file_lasttime: 上次处理中断的结果文件

    Returns:
        total_processed: 处理的样本总数
    """
    # 读取已处理的结果（用于跳过）
    processed_hashes = set()
    if processed_file_lasttime and os.path.exists(processed_file_lasttime):
        logger.info(f"读取上次处理结果: {processed_file_lasttime}")
        try:
            with open(processed_file_lasttime, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if text_field in data:
                            text_hash = hash(data[text_field])
                            processed_hashes.add(text_hash)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"发现 {len(processed_hashes)} 个已处理样本，将跳过这些样本")
        except Exception as e:
            logger.error(f"读取上次处理结果失败: {e}")

    total_processed = 0
    files_processed = set()

    # 如果存在上次处理结果，分析哪些文件已经处理过(有問題，如果上次處理一半程序停了，這次就不能跳過）
    # if processed_file_lasttime and os.path.exists(processed_file_lasttime):
    #     try:
    #         with open(processed_file_lasttime, 'r', encoding='utf-8') as f:
    #             for line in f:
    #                 try:
    #                     data = json.loads(line.strip())
    #                     if 'prediction_source' in data.get('prediction', {}):
    #                         source_file = data['prediction']['prediction_source']
    #                         files_processed.add(source_file)
    #                 except (json.JSONDecodeError, KeyError):
    #                     continue
    #         logger.info(f"发现 {len(files_processed)} 个已处理文件: {list(files_processed)}")
    #     except Exception as e:
    #         logger.error(f"分析已处理文件失败: {e}")

    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果不存在上次处理结果，清空输出文件
    # if not processed_file_lasttime or not os.path.exists(processed_file_lasttime):
    with open(output_file, 'w', encoding='utf-8') as f1, open('already_processed.jsonl', 'w', encoding='utf-8') as f2:
        f1.write("")
        f2.write("")


    for file_idx, jsonl_file in enumerate(jsonl_files):
        # 检查文件是否已经处理过
        file_basename = os.path.basename(jsonl_file)
        if file_basename in files_processed:
            logger.info(f"跳过已处理文件: {jsonl_file}")
            continue

        logger.info(f"流式处理文件 ({file_idx + 1}/{len(jsonl_files)}): {jsonl_file}")

        file_processed = 0
        file_total = 0
        file_duplicates = 0  # 记录文件内重复样本数

        # 使用流式读取处理文件
        for batch_idx, (batch_data, total_count) in enumerate(
                load_jsonl_file_streaming(jsonl_file, text_field, max_samples, read_batch_size)
        ):
            file_total = total_count  # 更新文件总计数

            # 提取文本，跳过已处理的
            batch_texts = []
            batch_original_data = []
            batch_duplicates = 0  # 当前批次重复样本数

            for data in batch_data:
                if text_field in data and data[text_field].strip():
                    text = data[text_field].strip()
                    if hash(text) not in processed_hashes:  # 跳过已处理的
                        # print(f"新样本:{text}, len(processed_hashes)={len(processed_hashes)}")
                        with open('already_processed.jsonl', 'a', encoding='utf-8') as f:
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')

                        batch_texts.append(text)
                        batch_original_data.append(data)
                    else:
                        batch_duplicates += 1

            file_duplicates += batch_duplicates  # 累加重复样本数

            if not batch_texts:
                logger.info(f"批次 {batch_idx + 1} 所有样本已处理，跳过")
                continue

            logger.info(f"处理批次 {batch_idx + 1}: {len(batch_texts)} 个新样本")

            # 为当前批次创建临时输出文件
            temp_output = f"{output_file}.{file_idx}.{batch_idx}.tmp"

            # 使用保证完成的方法处理当前批次
            processed_count = classifier.predict_with_guaranteed_completion(
                texts=batch_texts,
                output_file=temp_output,
                max_length=max_length,
                initial_batch_size=batch_size
            )

            # 读取预测结果并合并
            try:
                with open(temp_output, 'r', encoding='utf-8') as f:
                    predictions = [json.loads(line) for line in f]

                # 合并数据
                curr_results = []
                for i, (orig_data, pred) in enumerate(zip(batch_original_data, predictions)):
                    result = orig_data.copy()
                    result.update({
                        "prediction": {
                            "label": pred["label"],
                            "label_name": pred["label_name"],
                            "confidence": pred["confidence"],
                            "probabilities": pred["probabilities"],
                            "prediction_source": os.path.basename(jsonl_file),
                        },
                    })
                    curr_results.append(result)

                # 追加到输出文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    for result in curr_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                # 标记为已处理
                for text in batch_texts:
                    processed_hashes.add(hash(text))

                # 清理临时文件
                os.remove(temp_output)

                file_processed += processed_count
                total_processed += processed_count

            except Exception as e:
                logger.error(f"处理批次结果失败: {e}")

            # 验证当前批次处理完整性（在删除变量前进行）
            expected_processed = len(batch_texts)
            if processed_count != expected_processed:
                logger.warning(f"批次 {batch_idx + 1} 处理不完整: {processed_count}/{expected_processed}")

            # 清理内存，准备下一个批次
            del batch_data, batch_texts, batch_original_data
            if 'predictions' in locals():
                del predictions
            classifier._cleanup_memory(aggressive=True)

            logger.info(f"批次 {batch_idx + 1} 完成: {processed_count} 个新样本")

        # 验证文件处理完整性（使用正确的统计信息）
        expected_total_processed = file_total - file_duplicates
        if file_processed != expected_total_processed:
            logger.warning(f"文件 {jsonl_file} 处理不完整: {file_processed}/{expected_total_processed}")
        else:
            logger.info(
                f"文件 {jsonl_file} 处理完成: 总行数{file_total}, 已处理{file_processed}新样本, 跳过{file_duplicates}个重复样本")

    logger.info(f"所有文件处理完成: 共 {total_processed} 个新样本")
    return total_processed


def guarantee_process_all_files(classifier, jsonl_files: List[str], output_file: str,
                                text_field: str = "content", max_length: int = 512,
                                batch_size: int = 32, max_samples: int = None,
                                streaming: bool = True, processed_file_lasttime: str = None, read_batch_size: int = 6000):
    """
    保证处理所有文件的批量预测（支持流式和非流式模式）

    Args:
        streaming: 是否使用流式处理模式
        processed_file_lasttime: 上次处理中断的结果文件
    """
    if streaming:
        logger.info(f"使用流式处理数据，batch_size:{batch_size}, max_samples:{max_samples}")
        return guarantee_process_all_files_streaming(
            classifier, jsonl_files, output_file, text_field,
            max_length, batch_size, max_samples, read_batch_size,
            processed_file_lasttime=processed_file_lasttime
        )
    else:
        # 非流式模式（原有逻辑，增加断点续传支持）
        all_results = []
        processed_hashes = set()

        # 读取已处理结果
        if processed_file_lasttime and os.path.exists(processed_file_lasttime):
            logger.info(f"读取上次处理结果: {processed_file_lasttime}")
            try:
                with open(processed_file_lasttime, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if text_field in data:
                                text_hash = hash(data[text_field])
                                processed_hashes.add(text_hash)
                        except json.JSONDecodeError:
                            continue
                logger.info(f"发现 {len(processed_hashes)} 个已处理样本")
            except Exception as e:
                logger.error(f"读取上次处理结果失败: {e}")

        for file_idx, jsonl_file in enumerate(jsonl_files):
            logger.info(f"处理文件 ({file_idx + 1}/{len(jsonl_files)}): {jsonl_file}")

            # 读取文件
            texts, original_data = load_jsonl_file(jsonl_file, text_field, max_samples)

            if not texts:
                logger.warning(f"文件 {jsonl_file} 中没有有效数据")
                continue

            # 过滤已处理的文本
            new_texts = []
            new_original_data = []
            duplicates_count = 0
            for text, orig_data in zip(texts, original_data):
                if hash(text) not in processed_hashes:
                    new_texts.append(text)
                    new_original_data.append(orig_data)
                else:
                    duplicates_count += 1

            if not new_texts:
                logger.info(f"文件 {jsonl_file} 所有样本已处理，跳过")
                continue

            logger.info(f"文件 {jsonl_file} 有 {len(new_texts)} 个新样本需要处理")

            # 为每个文件创建临时输出
            temp_output = f"{output_file}.{file_idx}.tmp"

            # 使用保证完成的方法处理
            processed_count = classifier.predict_with_guaranteed_completion(
                texts=new_texts,
                output_file=temp_output,
                max_length=max_length,
                initial_batch_size=batch_size
            )

            # 验证处理结果
            if processed_count != len(new_texts):
                logger.error(f"文件 {jsonl_file} 处理不完整: {processed_count}/{len(new_texts)}")

            # 读取并合并结果
            try:
                with open(temp_output, 'r', encoding='utf-8') as f:
                    predictions = [json.loads(line) for line in f]

                # 合并数据
                for i, (orig_data, pred) in enumerate(zip(new_original_data, predictions)):
                    result = orig_data.copy()
                    result.update({
                        "model_pred": {
                            "label": pred["label"],
                            "label_name": pred["label_name"],
                            "confidence": pred["confidence"],
                            "probabilities": pred["probabilities"]
                        }
                    })
                    all_results.append(result)

                # 清理临时文件
                os.remove(temp_output)

            except Exception as e:
                logger.error(f"处理结果失败: {e}")

            logger.info(f"文件 {jsonl_file} 处理完成: {processed_count} 新样本, 跳过{duplicates_count}个重复样本")

        # 保存最终结果（如果是续传，需要合并旧结果）
        if processed_file_lasttime and os.path.exists(processed_file_lasttime):
            logger.info("合并上次处理结果...")
            try:
                with open(processed_file_lasttime, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line.strip()))
            except Exception as e:
                logger.error(f"合并上次结果失败: {e}")

        # 保存结果
        if all_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"所有结果保存到: {output_file}")

        return len(all_results)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="保证处理所有数据的分类器推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--text", type=str, help="要预测的单个文本")
    parser.add_argument("--text_file", type=str, help="文本文件路径")
    parser.add_argument("--jsonl_files", type=str, nargs='+', help="JSONL文件列表")
    parser.add_argument("--text_field", type=str, default="content", help="文本字段名")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--batch_size", type=int, default=32, help="初始批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max_memory_mb", type=int, default=1024 * 50, help="最大内存(MB)")
    parser.add_argument("--streaming", action="store_false", help="使用流式处理模式")
    parser.add_argument("--read_batch_size", type=int, default=5000, help="流式读取的批次大小")
    parser.add_argument("--processed_file_lasttime", type=str, help="上次处理中断的结果文件，用于断点续传")

    args = parser.parse_args()

    # 初始化分类器
    classifier = TextClassifier(args.model_path,
                                device=None if args.device == "auto" else args.device,
                                max_memory_mb=args.max_memory_mb)
    logger.info(f"本次任务 batch_size: {args.batch_size}")

    # 设置进程标题（如果可用）
    try:
        import setproctitle
        setproctitle.setproctitle(f"inference_{Path(args.model_path).stem}_batch{args.batch_size}_{os.getenv('LANG')}")
    except ImportError:
        pass

    # 处理数据
    if args.jsonl_files:
        # 批量处理JSONL文件
        processed_count = guarantee_process_all_files(
            classifier=classifier,
            jsonl_files=args.jsonl_files,
            output_file=args.output_file,
            text_field=args.text_field,
            max_length=args.max_length,
            batch_size=args.batch_size,
            streaming=args.streaming,
            processed_file_lasttime=args.processed_file_lasttime,
            read_batch_size=args.read_batch_size
        )
        logger.info(f"批量处理完成，共 {processed_count if processed_count else 0} 条结果")

    else:
        # 处理单个文本或文本文件
        if args.text_file:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        elif args.text:
            texts = [args.text]
        else:
            logger.error("请提供要预测的文本或文件")
            return

        # 使用保证完成的方法
        processed_count = classifier.predict_with_guaranteed_completion(
            texts=texts,
            output_file=args.output_file,
            max_length=args.max_length,
            initial_batch_size=args.batch_size,
            processed_file_lasttime=args.processed_file_lasttime
        )

        logger.info(f"处理完成: {processed_count} 个样本")


if __name__ == "__main__":
    main()