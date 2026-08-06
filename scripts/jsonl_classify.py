import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm


def classify_jsonl_by_label(
    input_path: str,
    output_dir: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, int]:
    """
    将JSONL文件按model_pred.label字段分类为neg和pos两类

    Args:
        input_path: 输入JSONL文件路径
        output_dir: 输出目录，默认为输入文件所在目录
        verbose: 是否显示详细信息

    Returns:
        包含分类统计信息的字典
    """
    # 处理路径
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 构建输出文件路径
    stem = input_path.stem
    neg_path = output_dir / f"{stem}_neg.jsonl"
    pos_path = output_dir / f"{stem}_pos.jsonl"

    # 统计信息
    stats = {
        "total": 0,
        "neg_count": 0,
        "pos_count": 0,
        "unknown_count": 0,
        "error_count": 0
    }

    if verbose:
        print(f"处理文件: {input_path}")
        print(f"输出目录: {output_dir}")

    with (
        open(input_path, 'r', encoding='utf-8') as f_in,
        open(neg_path, 'w', encoding='utf-8') as f_neg,
        open(pos_path, 'w', encoding='utf-8') as f_pos
    ):
        for line_num, line in enumerate(tqdm(f_in, desc="处理中"), 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                data = json.loads(line)

                # 获取label值
                # label = data.get("model_pred", {}).get("label", "")
                label = data.get("label", "")
                score = int(data.get("score", -1000000000))

                if label == "neg" and score <= 2:
                    f_neg.write(line + '\n')
                    stats["neg_count"] += 1
                elif label == "pos" and score >= 4:
                    f_pos.write(line + '\n')
                    stats["pos_count"] += 1
                # elif label == "__label_neg":  # 兼容可能的单下划线
                #     f_neg.write(line + '\n')
                #     stats["neg_count"] += 1
                # elif label == "__label_pos":  # 兼容可能的单下划线
                #     f_pos.write(line + '\n')
                #     stats["pos_count"] += 1
                else:
                    if verbose:
                        print(f"警告: 第{line_num}行未知label: {label}")
                    stats["unknown_count"] += 1

            except (json.JSONDecodeError, KeyError) as e:
                if verbose:
                    print(f"错误: 第{line_num}行解析失败: {e}")
                stats["error_count"] += 1

    return stats


def batch_classify_jsonl_files(
    input_pattern: str,
    output_dir: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    批量处理多个JSONL文件

    Args:
        input_pattern: 文件模式，支持通配符（如 "*.jsonl"）
        output_dir: 输出目录
        verbose: 是否显示详细信息

    Returns:
        包含每个文件统计信息的字典
    """
    from glob import glob

    all_stats = {}
    files = sorted(glob(input_pattern))

    if not files:
        print(f"警告: 没有找到匹配的文件: {input_pattern}")
        return all_stats

    print(f"找到 {len(files)} 个文件需要处理")

    for file_path in files:
        print(f"\n处理文件: {Path(file_path).name}")
        try:
            stats = classify_jsonl_by_label(file_path, output_dir, verbose)
            all_stats[file_path] = stats

            # 显示统计信息
            print(f"  总计: {stats['total']} 行")
            print(f"  neg: {stats['neg_count']} 行")
            print(f"  pos: {stats['pos_count']} 行")
            if stats['unknown_count'] > 0:
                print(f"  未知label: {stats['unknown_count']} 行")
            if stats['error_count'] > 0:
                print(f"  解析错误: {stats['error_count']} 行")

        except Exception as e:
            print(f"  处理失败: {e}")
            all_stats[file_path] = {"error": str(e)}

    return all_stats


def analyze_jsonl_file(input_path: str) -> Dict:
    """
    分析JSONL文件中的label分布

    Args:
        input_path: 输入文件路径

    Returns:
        包含分析结果的字典
    """
    input_path = Path(input_path)
    label_counts = {}
    label_examples = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                label = data.get("model_pred", {}).get("label", "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1

                # 保存每个label的一个例子
                if label not in label_examples:
                    label_examples[label] = data

            except json.JSONDecodeError:
                label_counts["parse_error"] = label_counts.get("parse_error", 0) + 1

    return {
        "total_lines": sum(label_counts.values()),
        "label_distribution": label_counts,
        "example_per_label": label_examples
    }


def main():
    parser = argparse.ArgumentParser(description='将JSONL文件按model_pred.label分类为neg和pos两类')
    parser.add_argument('input', help='输入文件或文件模式（如 "data/*.jsonl"）')
    parser.add_argument('-o', '--output', help='输出目录，默认为输入文件所在目录')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('-a', '--analyze', action='store_true', help='仅分析文件，不进行分类')
    parser.add_argument('-b', '--batch', action='store_true', help='批量处理多个文件')

    args = parser.parse_args()

    if args.analyze:
        # 分析模式
        if args.batch:
            from glob import glob
            files = sorted(glob(args.input))
            for file_path in files:
                print(f"\n分析文件: {file_path}")
                result = analyze_jsonl_file(file_path)
                print(f"  总行数: {result['total_lines']}")
                print("  label分布:")
                for label, count in result['label_distribution'].items():
                    print(f"    {label}: {count}")
        else:
            result = analyze_jsonl_file(args.input)
            print(f"分析结果: {Path(args.input).name}")
            print(f"总行数: {result['total_lines']}")
            print("label分布:")
            for label, count in result['label_distribution'].items():
                print(f"  {label}: {count}")

            print("\n示例数据:")
            for label, example in result['example_per_label'].items():
                print(f"  {label}: {example.get('text', '')[:100]}...")
    else:
        # 分类模式
        if args.batch:
            all_stats = batch_classify_jsonl_files(args.input, args.output, args.verbose)

            # 打印总体统计
            if all_stats:
                print("\n" + "="*50)
                print("处理完成！汇总统计:")

                total_stats = {
                    "total_files": len(all_stats),
                    "total_lines": 0,
                    "total_neg": 0,
                    "total_pos": 0,
                    "total_unknown": 0,
                    "total_errors": 0
                }

                for file_path, stats in all_stats.items():
                    if isinstance(stats, dict) and "error" not in stats:
                        total_stats["total_lines"] += stats.get("total", 0)
                        total_stats["total_neg"] += stats.get("neg_count", 0)
                        total_stats["total_pos"] += stats.get("pos_count", 0)
                        total_stats["total_unknown"] += stats.get("unknown_count", 0)
                        total_stats["total_errors"] += stats.get("error_count", 0)

                print(f"处理文件数: {total_stats['total_files']}")
                print(f"总行数: {total_stats['total_lines']}")
                print(f"neg分类: {total_stats['total_neg']}")
                print(f"pos分类: {total_stats['total_pos']}")
                if total_stats['total_unknown'] > 0:
                    print(f"未知label: {total_stats['total_unknown']}")
                if total_stats['total_errors'] > 0:
                    print(f"解析错误: {total_stats['total_errors']}")
        else:
            # 单个文件处理
            stats = classify_jsonl_by_label(args.input, args.output, args.verbose)

            print("\n" + "="*50)
            print("处理完成！")
            print(f"输入文件: {Path(args.input).name}")
            print(f"总行数: {stats['total']}")
            print(f"neg分类: {stats['neg_count']} (保存到: {Path(args.input).stem}_neg.jsonl)")
            print(f"pos分类: {stats['pos_count']} (保存到: {Path(args.input).stem}_pos.jsonl)")
            if stats['unknown_count'] > 0:
                print(f"未知label: {stats['unknown_count']}")
            if stats['error_count'] > 0:
                print(f"解析错误: {stats['error_count']}")


if __name__ == "__main__":
    main()