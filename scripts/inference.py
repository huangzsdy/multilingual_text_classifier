#!/usr/bin/env python3
"""
推理脚本 - 使用训练好的模型进行预测
支持单文本、文本文件、JSONL文件批量预测
"""

import os
import json
import torch
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassifier:
    """文本分类器"""
    
    def __init__(self, model_path: str, device=None):
        """
        初始化分类器
        
        Args:
            model_path: 模型路径
            device: 设备 (cpu/cuda)，如果为None则自动选择
        """
        self.model_path = model_path
        
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.load_model()
    
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
    
    def predict(self, texts: Union[str, List[str]], max_length=512, batch_size=32, return_probs=False):
        """
        对文本进行分类预测
        
        Args:
            texts: 单个文本或文本列表
            max_length: 最大序列长度
            batch_size: 批处理大小
            return_probs: 是否返回概率
            
        Returns:
            预测结果
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_predictions = []
        all_probabilities = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
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
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
        
        if return_probs:
            return all_predictions, all_probabilities
        return all_predictions
    
    def predict_with_label(self, texts: Union[str, List[str]], label_mapping=None, **kwargs):
        """
        预测并返回标签名称
        
        Args:
            texts: 文本
            label_mapping: 标签映射，如 {0: "负向", 1: "正向"}
            
        Returns:
            包含预测标签和概率的字典列表
        """
        if label_mapping is None:
            # 默认标签映射
            label_mapping = {0: "negative", 1: "positive"}
        
        predictions, probabilities = self.predict(texts, return_probs=True, **kwargs)
        
        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            result = {
                "text": text,
                "label": int(pred),
                "label_name": label_mapping.get(int(pred), f"label_{pred}"),
                "confidence": float(max(probs)),
                "probabilities": {label_mapping.get(i, f"label_{i}"): float(prob) 
                                 for i, prob in enumerate(probs)},
                'pos_prob' : probs[1]
            }
            results.append(result)
        
        return results

def load_jsonl_file(file_path: str, text_field: str = "content", max_samples: int = None):
    """
    从JSONL文件中读取数据
    
    Args:
        file_path: JSONL文件路径
        text_field: 包含文本的字段名，默认为"content"
        max_samples: 最大读取样本数（用于测试）
        
    Returns:
        texts: 文本列表
        original_data: 原始数据列表（包含所有字段）
    """
    texts = []
    original_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
            
            for i, line in enumerate(tqdm(lines, desc=f"读取 {os.path.basename(file_path)}")):
                try:
                    data = json.loads(line.strip())
                    
                    # 检查是否包含指定的文本字段
                    if text_field in data and data[text_field].strip():
                        texts.append(data[text_field].strip())
                        original_data.append(data)
                    else:
                        logger.warning(f"第 {i+1} 行缺少 '{text_field}' 字段或内容为空，已跳过")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {i+1} 行JSON解析错误: {e}")
                    continue
                    
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        raise
    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        raise
    
    logger.info(f"从 {file_path} 读取了 {len(texts)} 个有效样本")
    return texts, original_data

def predict_jsonl_files(classifier, jsonl_files: List[str], text_field: str = "content", 
                       batch_size: int = 32, max_length: int = 512, 
                       output_file: str = None, max_samples: int = None):
    """
    对JSONL文件进行批量预测
    
    Args:
        classifier: 文本分类器实例
        jsonl_files: JSONL文件路径列表
        text_field: 包含文本的字段名
        batch_size: 批处理大小
        max_length: 最大序列长度
        output_file: 输出文件路径
        max_samples: 最大样本数（用于测试）
        
    Returns:
        all_results: 所有预测结果
    """
    all_results = []
    
    for jsonl_file in jsonl_files:
        logger.info(f"处理文件: {jsonl_file}")
        
        # 读取JSONL文件
        texts, original_data = load_jsonl_file(jsonl_file, text_field, max_samples)
        
        if not texts:
            logger.warning(f"文件 {jsonl_file} 中没有有效数据，跳过")
            continue
        
        # 进行预测
        predictions = classifier.predict_with_label(
            texts, 
            batch_size=batch_size, 
            max_length=max_length
        )
        
        # 合并原始数据和预测结果
        for i, (orig_data, pred) in enumerate(zip(original_data, predictions)):
            # 添加预测结果到原始数据
            result = orig_data.copy()
            result.update({
                "model_pred": {
                    "label": pred["label"],
                    "label_name": pred["label_name"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "pos_prob" : pred['pos_prob']
                },
                "prediction_source": os.path.basename(jsonl_file),
                "prediction_timestamp": torch.tensor(0).to(classifier.device).cpu().numpy().tolist()  # 占位符
            })
            all_results.append(result)
        
        logger.info(f"文件 {jsonl_file} 预测完成: {len(predictions)} 个样本")
    
    # 保存结果到文件
    if output_file:
        save_predictions(all_results, output_file)
    
    return all_results

def save_predictions(results: List[dict], output_file: str):
    """
    保存预测结果到文件
    
    Args:
        results: 预测结果列表
        output_file: 输出文件路径
    """
    try:

        file_path = Path(output_file)

        # 一行搞定：先确保目录存在，然后创建文件
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"预测结果已保存到: {output_file}")
        logger.info(f"共保存 {len(results)} 条记录")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def load_config(config_path='config.json'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="XLM-RoBERTa分类器推理")
    parser.add_argument("--model_path", type=str, 
                       default="./results",
                       help="模型路径，默认为训练输出目录")
    parser.add_argument("--config", type=str, 
                       default="config.json",
                       help="配置文件路径")
    parser.add_argument("--text", type=str, 
                       help="要预测的单个文本")
    parser.add_argument("--text_file", type=str, 
                       help="包含多行文本的文件路径，每行一个文本")
    parser.add_argument("--jsonl_files", type=str, nargs='+',
                       help="一个或多个JSONL文件路径，进行批量预测")
    parser.add_argument("--text_field", type=str, 
                       default="content",
                       help="JSONL文件中包含文本的字段名，默认为'content'")
    parser.add_argument("--output_file", type=str, 
                       help="输出结果文件路径")
    parser.add_argument("--device", type=str, 
                       choices=["cpu", "cuda", "auto"],
                       default="auto",
                       help="运行设备")
    parser.add_argument("--batch_size", type=int, 
                       default=16,
                       help="批处理大小")
    parser.add_argument("--max_length", type=int, 
                       default=512,
                       help="最大序列长度")
    parser.add_argument("--max_samples", type=int, 
                       help="最大处理样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == "auto":
        device = None
    else:
        device = args.device
    
    # 加载模型
    classifier = TextClassifier(args.model_path, device)
    
    # 获取要预测的文本
    texts_to_predict = []
    original_data = None
    
    # 优先级：单个文本 > 文本文件 > JSONL文件
    if args.text:
        # 单个文本预测
        texts_to_predict = [args.text]
        logger.info(f"对单个文本进行预测: {args.text[:100]}...")
        
    elif args.text_file:
        # 文本文件预测
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        texts_to_predict.append(line)
            logger.info(f"从文本文件读取了 {len(texts_to_predict)} 个样本")
        except FileNotFoundError:
            logger.error(f"文件不存在: {args.text_file}")
            return
    
    elif args.jsonl_files:
        # JSONL文件批量预测
        logger.info(f"开始处理 {len(args.jsonl_files)} 个JSONL文件")
        
        # 验证文件存在性
        valid_files = []
        for file_path in args.jsonl_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"文件不存在，已跳过: {file_path}")
        
        if not valid_files:
            logger.error("没有有效的JSONL文件可供处理")
            return
        
        # 进行批量预测
        results = predict_jsonl_files(
            classifier=classifier,
            jsonl_files=valid_files,
            text_field=args.text_field,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_file=args.output_file,
            max_samples=args.max_samples
        )
        
        # 打印统计信息
        if results:
            label_counts = {}
            for result in results:
                label = result["model_pred"]["label_name"]
                label_counts[label] = label_counts.get(label, 0) + 1
            
            logger.info("批量预测统计:")
            for label, count in label_counts.items():
                percentage = count / len(results) * 100
                logger.info(f"  {label}: {count} 个 ({percentage:.1f}%)")
        
        return
    
    else:
        # 如果没有提供文本，则从标准输入读取
        logger.info("请输入要分类的文本（输入空行结束）:")
        while True:
            try:
                text = input().strip()
                if not text:
                    break
                texts_to_predict.append(text)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    if not texts_to_predict:
        logger.warning("没有提供要预测的文本")
        return
    
    logger.info(f"开始对 {len(texts_to_predict)} 个文本进行预测...")
    
    # 进行预测
    try:
        results = classifier.predict_with_label(
            texts_to_predict,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # 输出结果
        for i, result in enumerate(results):
            output = f"文本 {i+1}: {result['text'][:100]}...\n" \
                    f"  预测标签: {result['label_name']} (置信度: {result['confidence']:.4f})\n" \
                    f"  详细概率: {result['probabilities']}\n"
            print(output)
        
        # 保存到文件
        if args.output_file:
            # 如果是JSONL格式输出，保持一致性
            if args.output_file.endswith('.jsonl'):
                jsonl_results = []
                for result in results:
                    jsonl_results.append({
                        "text": result["text"],
                        "prediction": {
                            "label": result["label"],
                            "label_name": result["label_name"],
                            "confidence": result["confidence"],
                            "probabilities": result["probabilities"]
                        }
                    })
                
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for item in jsonl_results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                # 普通文本输出
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"结果已保存到: {args.output_file}")
        
        # 统计结果
        label_counts = {}
        for result in results:
            label = result['label_name']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("预测统计:")
        for label, count in label_counts.items():
            percentage = count / len(results) * 100
            logger.info(f"  {label}: {count} 个 ({percentage:.1f}%)")
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
