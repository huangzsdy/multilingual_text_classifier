import os
import json
import logging
import toml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tools.utils import load_config_with_references

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path='config.json'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.toml'):
            config = toml.loads(os.path.expandvars(f.read()))
            # config = load_config_with_references(config_path)
        else:
            config = json.load(f)
    
    # 确保目录存在
    os.makedirs(config['training_config']['output_dir'], exist_ok=True)
    os.makedirs(config['training_config']['logging_dir'], exist_ok=True)
    
    return config

def load_jsonl_samples(file_path, field_name, label, max_samples=None):
    """
    从单个JSONL文件中加载样本，只读取指定字段
    
    Args:
        file_path: 文件路径
        field_name: 要读取的字段名
        label: 样本标签
        max_samples: 最大样本数限制
        
    Returns:
        samples: 文本列表
        labels: 标签列表
        skipped_count: 跳过的样本数
    """
    samples = []
    labels = []
    skipped_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # 只读取指定字段
                    if field_name in data and data[field_name] and str(data[field_name]).strip():
                        text = str(data[field_name]).strip()
                        samples.append(text)
                        labels.append(label)
                    else:
                        skipped_count += 1
                        logger.debug(f"文件 {file_path} 第 {i+1} 行: 字段 '{field_name}' 不存在或为空")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"文件 {file_path} 第 {i+1} 行JSON解析错误: {e}")
                    skipped_count += 1
                    continue
                    
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        raise
        
    if skipped_count > 0:
        logger.info(f"文件 {file_path} 跳过了 {skipped_count} 个无效样本")
        
    return samples, labels

def load_multiple_jsonl_samples(file_configs, label, max_samples=None):
    """
    从多个JSONL文件中加载样本，每个文件只读取一个特定字段
    
    Args:
        file_configs: 文件配置列表，每个元素可以是:
                     - 字符串: 文件路径，使用默认字段 "content"
                     - 字典: {"path": "文件路径", "field": "字段名"}
        label: 样本标签
        max_samples: 最大样本数限制
        
    Returns:
        all_samples: 所有文本列表
        all_labels: 所有标签列表
    """
    all_samples = []
    all_labels = []
    total_skipped = 0
    
    # 处理单个配置的情况
    if not isinstance(file_configs, list):
        file_configs = [file_configs]
    
    for i, config in enumerate(file_configs):
        # 处理配置格式
        if isinstance(config, str):
            file_info_cont = config.split()

            assert len(file_info_cont) > 1, f"输入:{config} 必须不为空"
            field_name = file_info_cont[1]
            if field_name == 'normal':
                field_name = 'text'
            file_path = file_info_cont[-1]

        elif isinstance(config, dict):
            # 如果是字典，提取路径和字段
            file_path = config.get('path', '')
            field_name = config.get('field', 'content')
        else:
            logger.warning(f"配置格式错误，已跳过: {config}")
            continue
            
        if not file_path:
            logger.warning(f"文件路径为空，已跳过配置项 {i}")
            continue
            
        try:
            # 加载样本
            samples, labels = load_jsonl_samples(
                file_path, 
                field_name, 
                label, 
                max_samples
            )
            all_samples.extend(samples)
            all_labels.extend(labels)
            logger.info(f"从 {file_path} 加载了 {len(samples)} 个样本 (字段: '{field_name}')")
            
        except Exception as e:
            logger.warning(f"加载文件 {file_path} 失败: {e}")
            continue
    
    logger.info(f"总共从 {len(file_configs)} 个文件加载了 {len(all_samples)} 个样本 (标签: {label})")
    return all_samples, all_labels

def create_dataset(positive_configs, negative_configs, test_size=0.2, random_seed=42, max_samples=None):
    """创建训练和测试数据集，支持多文件和单个字段"""
    logger.info("开始加载数据集...")
    
    # 加载正样本（支持多文件和单个字段）
    positive_samples, positive_labels = load_multiple_jsonl_samples(
        positive_configs, 
        label=1,
        max_samples=max_samples
    )
    
    # 加载负样本（支持多文件和单个字段）
    negative_samples, negative_labels = load_multiple_jsonl_samples(
        negative_configs, 
        label=0,
        max_samples=max_samples
    )
    
    if not positive_samples:
        raise ValueError("正样本文件为空或格式不正确")
    if not negative_samples:
        raise ValueError("负样本文件为空或格式不正确")
    
    # 合并样本
    all_samples = positive_samples + negative_samples
    all_labels = positive_labels + negative_labels
    
    # 计算类别分布
    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    logger.info(f"数据集统计: 总样本数={len(all_samples)}, 正样本={pos_count}, 负样本={neg_count}")
    
    # 文件来源统计
    logger.info(f"正样本来源文件: {len(positive_configs)} 个")
    logger.info(f"负样本来源文件: {len(negative_configs)} 个")
    
    # 划分训练集和测试集
    if test_size > 0:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_samples, all_labels, 
            test_size=test_size, 
            stratify=all_labels,
            random_state=random_seed
        )
        
        logger.info(f"训练集: {len(train_texts)} 个样本")
        logger.info(f"测试集: {len(test_texts)} 个样本")
        
        return train_texts, test_texts, train_labels, test_labels
    else:
        logger.info(f"使用所有 {len(all_samples)} 个样本进行训练")
        return all_samples, [], all_labels, []

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def prepare_datasets(train_texts, test_texts, train_labels, test_labels, tokenizer, max_length=512):
    """准备数据集"""
    
    def tokenize_function(texts):
        return tokenizer(
            texts, 
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    logger.info("开始分词处理...")
    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts) if test_texts else None
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
            
        def __len__(self):
            return len(self.labels)
    
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels) if test_encodings else None
    
    logger.info(f"训练集创建完成: {len(train_dataset)} 个样本")
    if test_dataset:
        logger.info(f"测试集创建完成: {len(test_dataset)} 个样本")
    
    return train_dataset, test_dataset

def train_classifier(config):
    """主训练函数"""
    logger.info(f"开始训练{config['model_config']['model_name']}分类器...")
    
    # 设置随机种子
    set_seed(config['data_config']['random_seed'])
    
    # 加载配置
    model_config = config['model_config']
    data_config = config['data_config']
    training_config = config['training_config']
    
    # 创建数据集（支持多文件和单个字段）
    train_texts, test_texts, train_labels, test_labels = create_dataset(
        positive_configs=data_config['positive_sample_configs'],
        negative_configs=data_config['negative_sample_configs'],
        test_size=data_config['test_size'],
        random_seed=data_config['random_seed']
    )
    
    # 加载分词器
    logger.info(f"加载分词器: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    # 加载模型
    logger.info(f"加载预训练模型: {model_config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_labels'],
        trust_remote_code=True, weights_only=True
    )
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_datasets(
        train_texts, test_texts, train_labels, test_labels,
        tokenizer, data_config['max_length']
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        warmup_steps=training_config.get('warmup_steps', 500),
        weight_decay=training_config.get('weight_decay', 0.01),
        learning_rate=training_config.get('learning_rate', 2e-5),
        logging_dir=training_config.get('logging_dir', './logs'),
        eval_strategy=training_config.get('eval_strategy', 'epoch'),
        save_strategy=training_config.get('save_strategy', 'epoch'),
        save_total_limit=training_config.get('save_total_limit', 2),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'accuracy'),
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        report_to="none",
        logging_first_step=True,
        group_by_length=True,
        greater_is_better=True,
    )
    
    # 早停回调
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # 检查是否有之前的检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"从检查点恢复训练: {last_checkpoint}")
    
    # 开始训练
    logger.info("开始模型训练...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 评估模型
    if eval_dataset:
        logger.info("评估模型...")
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        logger.info(f"评估结果: {eval_metrics}")
    
    logger.info(f"训练完成！模型已保存到: {training_args.output_dir}")
    
    return trainer, metrics

if __name__ == "__main__":
    import argparse
    # import setproctitle
    parser = argparse.ArgumentParser(description='将JSONL文件按model_pred.label分类为neg和pos两类')
    parser.add_argument('-c', '--config', help='选用的config')
    parser.add_argument('-t', '--procTitle', help='进程title')
    # parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    # parser.add_argument('-a', '--analyze', action='store_true', help='仅分析文件，不进行分类')
    # parser.add_argument('-b', '--batch', action='store_true', help='批量处理多个文件')

    args = parser.parse_args()
    try:
        # 加载配置
        config = load_config(args.config)

        # setproctitle.setproctitle(args.procTitle)

        # 开始训练
        trainer, metrics = train_classifier(config)
        
        logger.info("=" * 50)
        logger.info("训练完成！")
        logger.info(f"最终模型保存在: {config['training_config']['output_dir']}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
