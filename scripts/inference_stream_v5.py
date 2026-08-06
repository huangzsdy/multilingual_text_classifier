import os
import json
import torch
import argparse
import logging
import time
import gc
import hashlib
from pathlib import Path
from typing import List, Dict, Union, Iterator, Tuple, Optional, Set
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiGPUClassifier:
    """多GPU并行分类器 - 支持多卡推理和断点续传"""

    def __init__(self, model_path: str, gpu_ids: Optional[List[int]] = None,
                 use_amp: bool = True, max_memory_mb: int = 1024 * 20):
        """
        初始化多GPU分类器

        Args:
            model_path: 模型路径
            gpu_ids: GPU设备ID列表，如[0,1,2]，为None时自动检测可用GPU
            use_amp: 是否使用混合精度推理
            max_memory_mb: 单卡最大内存限制（MB）
        """
        self.model_path = model_path
        self.use_amp = use_amp and torch.cuda.is_available()
        self.max_memory_mb = max_memory_mb

        # 设置GPU设备
        if gpu_ids is None and torch.cuda.is_available():
            self.gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, list) and torch.cuda.is_available():
            self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = []

        if self.gpu_ids:
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            logger.info(f"使用GPU设备: {self.gpu_ids}")
        else:
            self.device = torch.device("cpu")
            logger.info("使用CPU设备")

        logger.info(f"混合精度: {self.use_amp}")
        logger.info(f"单卡内存限制: {max_memory_mb} MB")

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载模型和分词器（支持多GPU）"""
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                padding_side='right'
            )

            # 根据是否使用混合精度选择模型精度
            if self.use_amp and self.device.type == 'cuda':
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype
            )

            # 多GPU并行
            if len(self.gpu_ids) > 1:
                logger.info(f"启用多GPU并行，使用 {len(self.gpu_ids)} 个GPU")
                self.model = torch.nn.DataParallel(
                    self.model,
                    device_ids=self.gpu_ids
                )
                self.model.to(f"cuda:{self.gpu_ids[0]}")
            elif self.device.type == 'cuda':
                self.model.to(self.device)

            self.model.eval()

            # 启用CUDA图优化
            # if self.device.type == 'cuda' and hasattr(torch, 'compile'):
            #     try:
            #         self.model = torch.compile(self.model)
            #         logger.info("已启用模型编译优化")
            #     except Exception as e:
            #         logger.warning(f"模型编译失败: {e}")

            logger.info(f"模型加载成功: {self.model_path}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def _get_text_hash(self, text: str) -> str:
        """获取文本哈希（用于去重）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _adjust_batch_size_for_multi_gpu(self, batch_size: int) -> int:
        """为多GPU调整批次大小，使其为GPU数量的整数倍"""
        if len(self.gpu_ids) > 1:
            # 确保批次大小是GPU数量的整数倍
            adjusted = max(1, (batch_size // len(self.gpu_ids)) * len(self.gpu_ids))
            if adjusted != batch_size:
                logger.info(f"调整批次大小: {batch_size} -> {adjusted} (适配 {len(self.gpu_ids)} 个GPU)")
            return adjusted
        return batch_size

    def _check_gpu_memory(self) -> bool:
        """检查GPU内存是否充足"""
        if self.device.type != 'cuda':
            return True

        for gpu_id in self.gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                cached = torch.cuda.memory_reserved(gpu_id) / 1024 / 1024

                if allocated > self.max_memory_mb * 0.9:  # 超过90%内存
                    logger.warning(f"GPU {gpu_id} 内存使用过高: {allocated:.1f}MB/{self.max_memory_mb}MB")
                    return False
            except Exception as e:
                logger.warning(f"检查GPU {gpu_id} 内存失败: {e}")

        return True

    def _clean_gpu_memory(self):
        """清理GPU内存"""
        if self.device.type == 'cuda':
            for gpu_id in self.gpu_ids:
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                except:
                    pass
            gc.collect()

    def batch_predict(self, texts: List[str], max_length: int = 512,
                      batch_size: int = 64) -> Tuple[List[int], List[List[float]]]:
        """
        批量预测（支持多GPU）

        Args:
            texts: 文本列表
            max_length: 最大序列长度
            batch_size: 总批次大小（多GPU时会在各卡间平分）

        Returns:
            (预测标签列表, 概率列表)
        """
        if not texts:
            return [], []

        # 为多GPU调整批次大小
        adjusted_batch_size = self._adjust_batch_size_for_multi_gpu(batch_size)

        all_predictions = []
        all_probabilities = []

        for i in range(0, len(texts), adjusted_batch_size):
            batch_texts = texts[i:i + adjusted_batch_size]

            # 检查GPU内存
            if not self._check_gpu_memory():
                self._clean_gpu_memory()
                time.sleep(1)  # 等待内存释放

            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False
            )

            # 将输入数据移到主GPU
            if self.device.type == 'cuda':
                inputs = {k: v.to(f"cuda:{self.gpu_ids[0]}") for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                # 处理多GPU输出
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits

                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

            # 收集结果
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

            # 清理内存
            del inputs, outputs, logits, probabilities, predictions

            # 定期清理GPU内存
            if i > 0 and i % (adjusted_batch_size * 10) == 0:
                self._clean_gpu_memory()

        return all_predictions, all_probabilities

    def predict_with_checkpoint(self, texts: List[str], output_file: str,
                                checkpoint_file: str, label_mapping: Optional[Dict] = None,
                                max_length: int = 512, batch_size: int = 64) -> Tuple[int, List[Dict]]:
        """
        带断点续传的预测

        Args:
            texts: 文本列表
            output_file: 输出文件
            checkpoint_file: 断点文件，记录已处理文本的哈希
            label_mapping: 标签映射
            max_length: 最大序列长度
            batch_size: 批次大小

        Returns:
            (实际处理的新样本数量, 处理结果列表)
        """
        if label_mapping is None:
            label_mapping = {0: "negative", 1: "positive"}

        # 加载已处理的哈希
        processed_hashes = self._load_processed_hashes(checkpoint_file)
        logger.info(f"已加载 {len(processed_hashes)} 个已处理样本的哈希")

        # 过滤已处理的文本
        new_texts = []
        new_indices = []
        skipped_count = 0

        for idx, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash not in processed_hashes:
                new_texts.append(text)
                new_indices.append(idx)
            else:
                skipped_count += 1

        logger.info(f"总样本: {len(texts)}, 新样本: {len(new_texts)}, 跳过: {skipped_count}")

        if not new_texts:
            logger.info("没有新样本需要处理")
            return 0, []

        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # 写入缓冲区
        write_buffer = []
        BUFFER_SIZE = 100

        start_time = time.time()
        processed_count = 0
        results = []

        # 批量处理新样本
        predictions, probabilities = self.batch_predict(
            new_texts, max_length, batch_size
        )

        # 将结果写入缓冲区和检查点
        with open(checkpoint_file, 'a', encoding='utf-8') as f_check:
            for i, (text, pred, probs, idx) in enumerate(
                    zip(new_texts, predictions, probabilities, new_indices)):

                result = {
                    "text": text,
                    "label": int(pred),
                    "label_name": label_mapping.get(int(pred), f"label_{pred}"),
                    "confidence": float(max(probs)),
                    "probabilities": {label_mapping.get(k, f"label_{k}"): float(v)
                                      for k, v in enumerate(probs)},
                    "sample_index": idx,
                    "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # 添加到结果列表
                results.append(result)

                # 添加到写入缓冲区
                write_buffer.append(json.dumps(result, ensure_ascii=False))

                # 记录到检查点
                text_hash = self._get_text_hash(text)
                f_check.write(text_hash + '\n')

                processed_count += 1

                # 批量写入
                if len(write_buffer) >= BUFFER_SIZE:
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        f_out.write('\n'.join(write_buffer) + '\n')
                    write_buffer.clear()

        # 写入剩余数据
        if write_buffer:
            with open(output_file, 'a', encoding='utf-8') as f_out:
                f_out.write('\n'.join(write_buffer) + '\n')

        # 统计信息
        total_time = time.time() - start_time
        samples_per_second = processed_count / total_time if total_time > 0 else 0

        logger.info(f"处理完成: {processed_count} 个新样本")
        logger.info(f"总时间: {total_time:.1f}秒, 速度: {samples_per_second:.1f} 样本/秒")

        return processed_count, results

    def _load_processed_hashes(self, checkpoint_file: str) -> Set[str]:
        """从检查点文件加载已处理的哈希"""
        processed_hashes = set()

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            processed_hashes.add(line)
                logger.info(f"从检查点加载 {len(processed_hashes)} 个已处理样本")
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
        else:
            # 创建检查点文件
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                f.write("")
            logger.info(f"创建新检查点文件: {checkpoint_file}")

        return processed_hashes


class StreamProcessor:
    """流式处理器 - 支持断点续传和多文件处理"""

    def __init__(self, read_batch_size: int = 100000):
        self.read_batch_size = read_batch_size

    def read_jsonl_streaming(self, file_path: str, text_field: str = "content",
                             max_samples: Optional[int] = None) -> Iterator[Tuple[List[Dict], int]]:
        """流式读取JSONL文件"""
        batch_data = []
        total_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_samples and total_count >= max_samples:
                        break

                    try:
                        data = json.loads(line.strip())
                        if text_field in data and data[text_field].strip():
                            batch_data.append(data)
                            total_count += 1

                            if len(batch_data) >= self.read_batch_size:
                                yield batch_data, total_count
                                batch_data = []
                    except json.JSONDecodeError:
                        continue

                if batch_data:
                    yield batch_data, total_count

        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            raise

    def process_file_with_checkpoint(self, classifier: MultiGPUClassifier,
                                     file_path: str, output_file: str,
                                     checkpoint_file: str, text_field: str = "content",
                                     max_length: int = 512, batch_size: int = 64,
                                     total_processed_before: int = 0) -> Dict[str, int]:
        """
        处理单个文件（支持断点续传）

        Args:
            total_processed_before: 之前已处理的样本总数（包括所有历史已处理样本）

        Returns:
            处理统计信息字典
        """
        stats = {
            "total_samples": 0,
            "processed_samples": 0,
            "skipped_samples": 0,
            "error_samples": 0
        }

        logger.info(f"处理文件: {file_path}")

        # 创建临时输出文件
        temp_output = f"{output_file}.{os.path.basename(file_path)}.tmp"

        # 流式读取和处理
        for batch_idx, (batch_data, total_count) in enumerate(
                self.read_jsonl_streaming(file_path, text_field)):

            stats["total_samples"] = total_count

            # 提取文本
            batch_texts = []
            batch_original_data = []

            for data in batch_data:
                if text_field in data and data[text_field].strip():
                    batch_texts.append(data[text_field].strip())
                    batch_original_data.append(data)

            if not batch_texts:
                continue

            # 处理当前批次
            try:
                processed_count, batch_results = classifier.predict_with_checkpoint(
                    texts=batch_texts,
                    output_file=temp_output,
                    checkpoint_file=checkpoint_file,
                    max_length=max_length,
                    batch_size=batch_size
                )

                stats["processed_samples"] += processed_count

                # 计算累计处理样本数
                current_processed_total = total_processed_before + stats["processed_samples"]

                # 输出当前处理进度
                logger.info(f"批次 {batch_idx + 1} 完成: 新增 {processed_count} 个样本, "
                            f"累计处理 {current_processed_total} 个样本")

                # 合并预测结果到最终输出
                self._merge_predictions(temp_output, output_file, batch_original_data, file_path)

                # 清理临时文件
                if os.path.exists(temp_output):
                    os.remove(temp_output)

            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
                stats["error_samples"] += len(batch_texts)

        return stats

    def _merge_predictions(self, temp_output: str, final_output: str,
                           original_data: List[Dict], source_file: str):
        """合并临时预测结果到最终输出"""
        if not os.path.exists(temp_output):
            return

        try:
            with open(temp_output, 'r', encoding='utf-8') as f_temp:
                predictions = [json.loads(line) for line in f_temp]

            with open(final_output, 'a', encoding='utf-8') as f_final:
                for orig_data, pred in zip(original_data[:len(predictions)], predictions):
                    result = orig_data.copy()
                    result["prediction"] = {
                        "label": pred["label"],
                        "label_name": pred["label_name"],
                        "confidence": pred["confidence"],
                        "probabilities": pred["probabilities"],
                        "source_file": os.path.basename(source_file),
                        "process_timestamp": time.time()
                    }
                    f_final.write(json.dumps(result, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"合并预测结果失败: {e}")


def get_total_processed_from_checkpoints(checkpoint_dir: str) -> int:
    """
    从检查点目录获取已处理的样本总数

    Args:
        checkpoint_dir: 检查点目录路径

    Returns:
        已处理的样本总数
    """
    total_processed = 0

    if not os.path.exists(checkpoint_dir):
        return total_processed

    try:
        # 遍历检查点目录中的所有检查点文件
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.checkpoint.txt'):
                checkpoint_file = os.path.join(checkpoint_dir, filename)

                # 计算检查点文件的行数（每个行对应一个已处理样本的哈希）
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        line_count = 0
                        for line in f:
                            if line.strip():  # 跳过空行
                                line_count += 1
                        total_processed += line_count
                except Exception as e:
                    logger.warning(f"读取检查点文件失败 {checkpoint_file}: {e}")

        logger.info(f"从检查点目录统计到 {total_processed} 个已处理样本")
    except Exception as e:
        logger.error(f"统计检查点目录失败: {e}")

    return total_processed


def process_files_with_checkpoints(classifier: MultiGPUClassifier,
                                   jsonl_files: List[str], output_file: str,
                                   checkpoint_dir: str, text_field: str = "content",
                                   max_length: int = 512, batch_size: int = 64) -> Dict[str, int]:
    """
    处理多个文件（支持断点续传）

    Args:
        classifier: 分类器实例
        jsonl_files: JSONL文件列表
        output_file: 最终输出文件
        checkpoint_dir: 检查点目录
        text_field: 文本字段名
        max_length: 最大序列长度
        batch_size: 批次大小

    Returns:
        总体统计信息
    """
    # 确保目录存在
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # 清空或创建最终输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("")

    # 初始化处理器
    processor = StreamProcessor(read_batch_size=10000)

    # 统计历史已处理的样本总数
    total_processed_before = get_total_processed_from_checkpoints(checkpoint_dir)
    logger.info(f"历史已处理样本总数: {total_processed_before}")

    total_stats = {
        "total_files": len(jsonl_files),
        "processed_files": 0,
        "total_samples": 0,
        "processed_samples": 0,
        "skipped_samples": 0,
        "error_samples": 0,
        "historical_processed_samples": total_processed_before
    }

    start_time = time.time()

    for file_idx, jsonl_file in enumerate(jsonl_files):
        logger.info(f"处理文件 ({file_idx + 1}/{len(jsonl_files)}): {jsonl_file}")

        # 为每个文件创建独立的检查点
        file_basename = os.path.basename(jsonl_file)
        checkpoint_file = os.path.join(checkpoint_dir, f"{file_basename}.checkpoint.txt")

        # 处理文件
        file_stats = processor.process_file_with_checkpoint(
            classifier=classifier,
            file_path=jsonl_file,
            output_file=output_file,
            checkpoint_file=checkpoint_file,
            text_field=text_field,
            max_length=max_length,
            batch_size=batch_size,
            total_processed_before=total_processed_before
        )

        # 更新统计
        total_stats["processed_files"] += 1
        total_stats["total_samples"] += file_stats["total_samples"]
        total_stats["processed_samples"] += file_stats["processed_samples"]
        total_stats["skipped_samples"] += file_stats["skipped_samples"]
        total_stats["error_samples"] += file_stats["error_samples"]

        logger.info(f"文件处理统计: 总计{file_stats['total_samples']}, "
                    f"处理{file_stats['processed_samples']}, "
                    f"跳过{file_stats['skipped_samples']}, "
                    f"错误{file_stats['error_samples']}")

    # 最终统计
    total_time = time.time() - start_time

    # 重新统计最终已处理样本总数
    final_historical_processed = get_total_processed_from_checkpoints(checkpoint_dir)
    total_processed_this_run = total_stats["processed_samples"]
    total_processed_all_time = final_historical_processed

    total_stats["total_time_seconds"] = total_time
    total_stats["total_processed_this_run"] = total_processed_this_run
    total_stats["total_processed_all_time"] = total_processed_all_time

    if total_processed_this_run > 0:
        samples_per_second = total_processed_this_run / total_time
        total_stats["samples_per_second"] = samples_per_second
    else:
        total_stats["samples_per_second"] = 0

    logger.info(f"所有文件处理完成")
    logger.info(f"本次运行处理样本: {total_stats['processed_samples']}, "
                f"跳过: {total_stats['skipped_samples']}, "
                f"错误: {total_stats['error_samples']}")
    logger.info(f"历史累计处理样本: {total_stats['historical_processed_samples']}")
    logger.info(f"累计总处理样本: {total_processed_all_time}")
    logger.info(f"总时间: {total_time:.1f}秒, "
                f"本次运行速度: {total_stats['samples_per_second']:.1f} 样本/秒")

    return total_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多GPU并行文本分类推理（支持断点续传和累计统计）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--jsonl_files", type=str, nargs='+', required=True, help="JSONL文件列表")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点目录，用于断点续传")
    parser.add_argument("--text_field", type=str, default="content", help="文本字段名")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="GPU设备ID，用逗号分隔，如 '0,1,2'，为空时自动检测可用GPU")
    parser.add_argument("--batch_size", type=int, default=64, help="每个GPU的批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度推理")
    parser.add_argument("--max_memory_mb", type=int, default=1024 * 20,
                        help="单卡最大内存限制（MB）")

    args = parser.parse_args()

    # 解析GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    # 初始化多GPU分类器
    classifier = MultiGPUClassifier(
        model_path=args.model_path,
        gpu_ids=gpu_ids,
        use_amp=args.use_amp,
        max_memory_mb=args.max_memory_mb
    )

    logger.info(f"开始处理 {len(args.jsonl_files)} 个文件")
    logger.info(f"输出文件: {args.output_file}")
    logger.info(f"检查点目录: {args.checkpoint_dir}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"GPU数量: {len(classifier.gpu_ids) if classifier.gpu_ids else 1}")

    # 处理所有文件
    stats = process_files_with_checkpoints(
        classifier=classifier,
        jsonl_files=args.jsonl_files,
        output_file=args.output_file,
        checkpoint_dir=args.checkpoint_dir,
        text_field=args.text_field,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    # 输出最终统计
    print("\n" + "=" * 60)
    print("处理完成统计:")
    print(f"处理文件: {stats['processed_files']}/{stats['total_files']}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"本次运行处理样本: {stats['processed_samples']}")
    print(f"本次运行跳过样本: {stats['skipped_samples']}")
    print(f"本次运行错误样本: {stats['error_samples']}")
    print(f"历史累计处理样本: {stats['historical_processed_samples']}")
    print(f"累计总处理样本: {stats['total_processed_all_time']}")
    print(f"总时间: {stats['total_time_seconds']:.1f}秒")
    print(f"本次运行处理速度: {stats['samples_per_second']:.1f} 样本/秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
