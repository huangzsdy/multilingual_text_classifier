import json
import os
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import pandas as pd


def calculate_confusion_matrix(true_labels: List, pred_labels: List) -> Dict[str, Dict[str, int]]:
    """
    计算混淆矩阵

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表

    Returns:
        混淆矩阵字典: {真实标签: {预测标签: 数量}}
    """
    confusion = defaultdict(lambda: defaultdict(int))

    for true, pred in zip(true_labels, pred_labels):
        confusion[str(true)][str(pred)] += 1

    return dict(confusion)


def calculate_metrics(
        true_labels: List[str],
        pred_labels: List[str]
) -> Dict[str, float]:
    """
    计算分类评估指标（纯Python实现，不依赖scikit-learn）

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表

    Returns:
        评估指标字典
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("真实标签和预测标签数量不一致")

    if not true_labels or not pred_labels:
        return {}

    n_samples = len(true_labels)

    # 1. 计算准确率
    correct = 0
    for true, pred in zip(true_labels, pred_labels):
        if str(true) == str(pred):
            correct += 1
    accuracy = correct / n_samples if n_samples > 0 else 0.0

    # 2. 获取所有标签
    all_labels = sorted(set([str(label) for label in true_labels + pred_labels]))

    # 3. 计算每个类别的统计
    per_class_stats = {}
    print(f"all_labels: {all_labels}")
    for label in all_labels:
        tp = fp = fn = 0

        for true, pred in zip(true_labels, pred_labels):
            true_str = str(true)
            pred_str = str(pred)

            if true_str == label and pred_str == label:
                tp += 1
            elif true_str != label and pred_str == label:
                fp += 1
            elif true_str == label and pred_str != label:
                fn += 1

        # 计算精确率、召回率、F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # print(f"precision: {precision}, recall: {recall}, f1: {f1}")
        support = tp + fn  # 真实样本数

        per_class_stats[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': support
        }

    # 4. 计算宏平均
    macro_precision = sum(stat['precision'] for stat in per_class_stats.values()) / len(
        per_class_stats) if per_class_stats else 0.0
    macro_recall = sum(stat['recall'] for stat in per_class_stats.values()) / len(
        per_class_stats) if per_class_stats else 0.0
    macro_f1 = sum(stat['f1'] for stat in per_class_stats.values()) / len(per_class_stats) if per_class_stats else 0.0
    print(f"macro_f1: {macro_f1}, {[stat['f1'] for stat in per_class_stats.values()]}")
    # 5. 计算微平均
    total_tp = sum(stat['tp'] for stat in per_class_stats.values())
    total_fp = sum(stat['fp'] for stat in per_class_stats.values())
    total_fn = sum(stat['fn'] for stat in per_class_stats.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) > 0 else 0.0

    # 6. 计算加权平均
    total_support = sum(stat['support'] for stat in per_class_stats.values())
    weighted_precision = sum(stat['precision'] * stat['support'] for stat in
                             per_class_stats.values()) / total_support if total_support > 0 else 0.0
    weighted_recall = sum(stat['recall'] * stat['support'] for stat in
                          per_class_stats.values()) / total_support if total_support > 0 else 0.0
    weighted_f1 = sum(
        stat['f1'] * stat['support'] for stat in per_class_stats.values()) / total_support if total_support > 0 else 0.0

    # 7. 计算混淆矩阵
    confusion_matrix = calculate_confusion_matrix(true_labels, pred_labels)

    # 8. 构建结果字典
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'n_samples': n_samples,
        'n_classes': len(all_labels)
    }

    # 添加每个类别的指标
    for label, stats in per_class_stats.items():
        metrics[f'precision_{label}'] = stats['precision']
        metrics[f'recall_{label}'] = stats['recall']
        metrics[f'f1_{label}'] = stats['f1']
        metrics[f'support_{label}'] = stats['support']

    # 添加混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix

    return metrics


def parse_jsonl_and_evaluate(
        result_json_list,
        label_mapping: Dict[str, str] = None
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """
    解析jsonl list并计算评估指标（不依赖scikit-learn）

    Args:
        result_json_list: jsonl的list
        label_mapping: 标签映射字典，用于将原始标签映射为标准标签

    Returns:
        Tuple[List[Dict], Dict]: 包含(content, label)的字典列表 和 评估指标字典
    """

    # 读取和解析jsonl文件
    content_label_pairs = []
    true_labels = []
    pred_labels = []

    for data in result_json_list:
        if not data:
            continue

        try:
            # 解析content
            content = data.get('content', '')
            url = data.get('url')
            # 解析真实标签
            true_label = None
            if 'label' in data:
                true_label = data['label']
                # true_label = str(data['score'])
                # print(f"true_label: {true_label}")
            # 解析预测标签
            pred_label = None
            if 'model_pred' in data and isinstance(data['model_pred'], dict):
                pred_label = data['model_pred'].get('label')
                # score = int(data['model_pred'].get('score'))
                # pred_label = str(score)
                # print(f"pred_label: {pred_label}")
            elif 'pred_label' in data:
                pred_label = data['pred_label']

            # 如果两者都有，才添加到评估数据中
            if true_label is not None and pred_label is not None:
                # 应用标签映射（如果需要）
                if label_mapping:
                    ori_true_label = true_label
                    # true_label = label_mapping.get(str(true_label), '正面')
                    # pred_label = label_mapping.get(str(pred_label), '正面')

                    # true_label = label_mapping.get(str(true_label), '负面')
                    # pred_label = label_mapping.get(str(pred_label), '负面')

                    true_label = label_mapping.get(str(true_label), str(true_label))
                    pred_label = label_mapping.get(str(pred_label), str(pred_label))
                    # print(f"trans label:ori_true_label:{ori_true_label}, true_label:{true_label}, pred_label:{pred_label}")
                content_label_pairs.append({
                    'url': url,
                    'content': content,
                    'true_label': true_label,
                    'pred_label': pred_label
                })

                true_labels.append(true_label)
                pred_labels.append(pred_label)
            # 如果只有真实标签，也添加到内容对中（但不用于评估）
            elif true_label is not None:
                if label_mapping:
                    true_label = label_mapping.get(str(true_label), str(true_label))

                content_label_pairs.append({
                    'content': content,
                    'label': true_label
                })

        except Exception as e:
            print(f"解析JSON行时出错: {e}")
            continue

    # 计算评估指标（如果有预测标签）
    metrics = {}
    if true_labels and pred_labels and len(true_labels) == len(pred_labels):
        metrics = calculate_metrics(true_labels, pred_labels)
    elif true_labels and not pred_labels:
        print("警告: 未找到预测标签，只返回内容-标签对")
    elif true_labels and pred_labels and len(true_labels) != len(pred_labels):
        print("警告: 真实标签和预测标签数量不一致，跳过指标计算")

    return content_label_pairs, metrics


def save_content_label_pairs(
        pairs: List[Dict[str, str]],
        output_path: str
) -> None:
    """
    保存content-label对到文件

    Args:
        pairs: content-label对列表
        output_path: 输出文件路径
    """
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write('\n')


def print_detailed_report(metrics: Dict, output_file: str = None, digits: int = 4):
    """
    打印详细的分类报告，并可选择保存到文件

    Args:
        metrics: 评估指标字典
        output_file: 输出文件路径（可选）
        digits: 小数位数
    """
    # return

    if not metrics:
        print("无评估指标可用")
        return

    # 构建输出内容
    lines = []

    lines.append("=" * 60)
    lines.append("分类评估报告")
    lines.append("=" * 60)

    # 准确率信息
    lines.append(f"准确率 (Accuracy): {metrics.get('accuracy', 0):.{digits}f}")
    lines.append(f"样本总数: {metrics.get('n_samples', 0)}")
    lines.append(f"类别总数: {metrics.get('n_classes', 0)}")
    lines.append("")

    # 各类别详细指标
    if any(key.startswith('precision_') for key in metrics.keys()):
        lines.append("各类别详细指标:")
        lines.append("-" * 60)
        lines.append(f"{'类别':<15} {'精确率':<10} {'召回率':<8} {'F1分数':<8} {'支持数':<10}")
        lines.append("-" * 60)

        class_metrics = []
        for key in metrics.keys():
            if key.startswith('precision_'):
                label = key.replace('precision_', '')
                class_metrics.append((
                    label,
                    metrics.get(f'precision_{label}', 0),
                    metrics.get(f'recall_{label}', 0),
                    metrics.get(f'f1_{label}', 0),
                    metrics.get(f'support_{label}', 0)
                ))

        # 按类别名排序
        class_metrics.sort(key=lambda x: x[0])

        for label, prec, rec, f1, sup in class_metrics:
            lines.append(f"{label:<15} {prec:.{digits}f}   {rec:.{digits}f}   {f1:.{digits}f}   {sup:<10}")

        lines.append("-" * 60)
        lines.append("")

    # 平均指标
    lines.append("平均指标:")
    lines.append(f"  宏平均精确率: {metrics.get('macro_precision', 0):.{digits}f}")
    lines.append(f"  宏平均召回率: {metrics.get('macro_recall', 0):.{digits}f}")
    lines.append(f"  宏平均F1: {metrics.get('macro_f1', 0):.{digits}f}")
    lines.append(f"  微平均精确率: {metrics.get('micro_precision', 0):.{digits}f}")
    lines.append(f"  微平均召回率: {metrics.get('micro_recall', 0):.{digits}f}")
    lines.append(f"  微平均F1: {metrics.get('micro_f1', 0):.{digits}f}")
    lines.append(f"  加权精确率: {metrics.get('weighted_precision', 0):.{digits}f}")
    lines.append(f"  加权召回率: {metrics.get('weighted_recall', 0):.{digits}f}")
    lines.append(f"  加权F1: {metrics.get('weighted_f1', 0):.{digits}f}")

    # 混淆矩阵
    if 'confusion_matrix' in metrics:
        lines.append("")
        lines.append("混淆矩阵:")
        cm = metrics['confusion_matrix']
        all_labels = sorted(set([k for row in cm.values() for k in row.keys()] + list(cm.keys())))

        header = "真实\\预测" + "".join([f"{label:>8}" for label in all_labels])
        lines.append(header)
        lines.append("-" * len(header))

        for true_label in all_labels:
            row = f"{true_label:^10}"
            for pred_label in all_labels:
                count = cm.get(str(true_label), {}).get(str(pred_label), 0)
                row += f"{count:>8}"
            lines.append(row)

    lines.append("=" * 60)

    # 输出到控制台
    for line in lines:
        print(line)

    # 保存到文件（如果指定了输出文件）
    if output_file:
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            print(f"\n报告已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")


def parse_result(save_results_path: str, result_jsonl_list: Optional[Dict] = None) -> None:
    """
    解析结果并生成报告

    Args:
        save_results_path: 结果保存路径
        result_jsonl_list: JSONL数据列表
    """
    if not result_jsonl_list:
        # 创建测试数据（多类别示例）
        test_data = [
            {
                "content": "这是一条体育新闻，比赛很精彩",
                "label": "体育",
                "model_pred": {"label": "体育", "score": 0.9}
            },
            {
                "content": "这是一条政治新闻，国际关系会谈",
                "label": "政治",
                "model_pred": {"label": "政治", "score": 0.8}
            },
            {
                "content": "这是一条娱乐新闻，新电影上映",
                "label": "娱乐",
                "model_pred": {"label": "体育", "score": 0.6}  # 错误预测
            },
            {
                "content": "这是一条经济新闻，股市行情分析",
                "label": "经济",
                "model_pred": {"label": "经济", "score": 0.7}
            },
            {
                "content": "这是一条科技新闻，人工智能发展",
                "label": "科技",
                "model_pred": {"label": "科技", "score": 0.9}
            }
        ]
        result_jsonl_list = test_data

    # 标签映射
    label_mapping = {
        # "正样本": "正面",
        # "正樣本": "正面",
        # "負樣本": "负面",
        # "负样本": "负面",
        # '負样本': "负面",
        # "负樣本": "负面",
        "__label__pos": "正面",
        "__label__neg": "负面",
        "positive": "正面",
        "negative": "负面",
        "pos": "正面",
        "neg": "负面",
        '0' : '负面',
        '1' : '正面'

        # '0' : '负面',
        # '1' : '负面',
        # "2": "负面",
        # "3": "正面",
        # "4": "正面",
        # "5": "正面",
    }
    # label_mapping = {}
    # 解析并评估
    try:
        pairs, metrics = parse_jsonl_and_evaluate(result_jsonl_list, label_mapping)

        print(f"解析到 {len(pairs)} 条数据:")
        error_count = 0
        for i, pair in enumerate(pairs):
            if 'true_label' in pair and 'pred_label' in pair:
                if pair['true_label'] != pair['pred_label']:
                    error_count += 1
                    # content_preview = pair['content'][:50] + "..." if len(pair['content']) > 50 else pair['content']
                    content_preview = pair['content']
                    # print(f"  第{i + 1}条错误预测:")
                    # print(f"    内容: {content_preview}")
                    # print(f"    真实标签={pair['true_label']}, 预测标签={pair['pred_label']}")
                    # print()

        print(f"总错误数: {error_count}/{len(pairs)}")

        # 生成详细报告
        report_path = os.path.join(save_results_path, "detailed_report.txt")
        print_detailed_report(metrics, report_path)

        # 保存结果
        save_output = os.path.join(save_results_path, "content_label_pairs.jsonl")
        save_content_label_pairs(pairs, save_output)
        print(f"\n结果已保存到: {save_output}")

    except Exception as e:
        print(f"处理JSONL数据时出错: {e}")
        traceback.print_exc()


def read_jsonl(file_path: str) -> List[Dict]:
    """
    读取JSONL文件

    Args:
        file_path: 文件路径

    Returns:
        JSON数据列表
    """
    path = Path(file_path)
    if not path.exists():
        print(f"文件不存在: {file_path}")
        return []

    file_size = path.stat().st_size
    print(f"开始读取文件 {file_path} ({file_size / 1024 / 1024:.2f} MB)")

    data_list = []
    start_time = time.time()

    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    data_list.append(json.loads(line.strip()))

                if i % 5000 == 0:
                    elapsed = time.time() - start_time
                    progress = (f.tell() / file_size) * 100 if file_size > 0 else 0
                    print(f"进度: {progress:.1f}% ({i}行) - 耗时: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"完成！共读取 {len(data_list)} 条数据，耗时 {total_time:.1f}s")

    except Exception as e:
        print(f"读取文件时出错: {e}")

    return data_list


def calculate_predictions(pred_file: str, ground_truth_file: str, save_results_path: str) -> None:
    """
    计算预测结果

    Args:
        pred_file: 预测结果文件路径
        ground_truth_file: 真实标签文件路径
        save_results_path: 结果保存路径
    """
    # 读取预测结果和真实标签
    pred_cont = read_jsonl(pred_file)
    gt_cont = read_jsonl(ground_truth_file)

    if not pred_cont or not gt_cont:
        print("错误: 无法读取输入文件")
        return

    # # 合并pred和gt
    # pred_temp_dict = {item['url']: item for item in pred_cont}
    #
    # # 在gt的json结构中加入model_pred字段
    # merge_dict = [
    #     {**item, 'model_pred': pred_temp_dict[item['url']]['model_pred']}
    #     for item in gt_cont if item['url'] in pred_temp_dict
    # ]

    # 合并pred和gt
    pred_temp_dict = {item['content']: item for item in pred_cont}
    # ours
    # if 'zh_labeled.jsonl' in pred_file or 'zh.jsonl' in pred_file:
    if True:
        merge_dict = [
            # {**item, "model_pred": {"label": pred_temp_dict[item['content']]['pred_label'],
        {**item, "model_pred": {"label": pred_temp_dict[item['content']]['model_pred']['label'],
                                    "score": pred_temp_dict[item['content']]['score']}}
            for item in gt_cont if item['content'] in pred_temp_dict
        ]
    # else:
    #     # 在gt的json结构中加入model_pred字段
    #     merge_dict = [
    #         {**item, "model_pred": {"label": pred_temp_dict[item['content']]['pred_label'],
    #         # {**item, "model_pred": {"label": pred_temp_dict[item['content']]['model_pred']['label'],
    #                                                         "score": pred_temp_dict[item['content']]['score']}}
    #         for item in gt_cont if item['content'] in pred_temp_dict
    #     ]


    print(f"成功合并 {len(merge_dict)} 条数据")

    # 解析结果
    parse_result(save_results_path, merge_dict)


def test_multiclass_classification():
    """
    测试多类别分类功能
    """
    print("测试多类别分类评估:")
    print("=" * 50)

    # 测试数据（5个类别）
    true_labels = ["体育", "政治", "娱乐", "经济", "科技", "体育", "政治", "娱乐"]
    pred_labels = ["体育", "政治", "体育", "经济", "科技", "娱乐", "政治", "娱乐"]  # 有2个错误

    print(f"真实标签: {true_labels}")
    print(f"预测标签: {pred_labels}")
    print()

    # 计算指标
    metrics = calculate_metrics(true_labels, pred_labels)
    print(f"metrics:{metrics}")
    print_detailed_report(metrics)

    # 显示每个类别的F1分数
    print("\n各类别F1分数:")
    for key in sorted(metrics.keys()):
        if key.startswith('f1_'):
            label = key.replace('f1_', '')
            f1_score = metrics[key]
            support = metrics.get(f'support_{label}', 0)
            print(f"  {label}: {f1_score:.4f} (支持数: {support})")


# 使用示例
if __name__ == "__main__":
    # 创建保存结果的目录
    save_results_path = './temp_save_results_path'
    if not os.path.isdir(save_results_path):
        os.makedirs(save_results_path)

    # 1. 测试多类别分类功能
    print("1. 测试多类别分类评估:")
    test_multiclass_classification()

    print("\n" + "=" * 80 + "\n")

    # 2. 测试从文件读取（如果文件存在）
    # pred_file = '/data/multilingual_projects/text_classification/output/minimal_batch_output/zh/final_pred/zh_10k.jsonl'
    # ground_truth_file = '/data/multilingual_projects/datasets/sample_data/val_labeled_qwen/zh_labeled.jsonl'
    #
    # # 检查文件是否存在，如果存在则处理，否则跳过
    # if os.path.exists(pred_file) and os.path.exists(ground_truth_file):
    #     print("2. 处理实际数据文件:")
    #     calculate_predictions(pred_file, ground_truth_file, save_results_path)
    # else:
    #     print("2. 文件不存在，跳过实际数据处理")
    #     print("使用测试数据进行演示:")
    #
    #     # 使用测试数据演示
    #     test_data = [
    #         {
    #             "url": "http://example.com/1",
    #             "content": "篮球比赛非常精彩，球员表现出色",
    #             "label": "体育"
    #         },
    #         {
    #             "url": "http://example.com/2",
    #             "content": "国际政治会谈取得重要进展",
    #             "label": "政治"
    #         },
    #         {
    #             "url": "http://example.com/3",
    #             "content": "新电影票房创下纪录",
    #             "label": "娱乐"
    #         }
    #     ]
    #
    #     # 添加预测结果
    #     for i, item in enumerate(test_data):
    #         if i == 2:  # 第三个样本预测错误
    #             item['model_pred'] = {"label": "体育", "score": 0.6}
    #         else:
    #             item['model_pred'] = {"label": item['label'], "score": 0.9}
    #
    #     parse_result(save_results_path, test_data)