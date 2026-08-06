import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys


def read_and_deduplicate_jsonl_files(file_paths: List[str], key_fields: List[str] = ["text", "content"]) -> List[Dict]:
    """
    读取多个JSONL文件，按多个字段优先级去重

    Args:
        file_paths: JSONL文件路径列表
        key_fields: 用于去重的字段名列表（按优先级排序）

    Returns:
        去重后的数据列表（统一使用content字段）
    """
    seen_contents = set()
    deduplicated_data = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # 按优先级查找可用的去重字段
                        content_value = None
                        used_field = None

                        for field in key_fields:
                            if field in data:
                                content_value = data[field]
                                used_field = field
                                break

                        # 如果找到可用的字段
                        if content_value is not None:
                            # 如果内容未出现过，添加到结果
                            if content_value not in seen_contents:
                                seen_contents.add(content_value)

                                # 创建新记录，统一使用content字段
                                new_data = data.copy()

                                # 如果使用的不是content字段，需要特殊处理
                                if used_field != "content":
                                    # 保留原content字段（如果存在）
                                    if "content" in new_data and used_field != "content":
                                        # 可以选择保留原content到其他字段，或者覆盖
                                        # 这里选择覆盖，将原content保存到original_content
                                        if "original_content" not in new_data:
                                            new_data["original_content"] = new_data.get("content")

                                    # 统一设置content字段
                                    new_data["content"] = content_value

                                # 添加来源信息（可选）
                                if "source_field" not in new_data:
                                    new_data["source_field"] = used_field

                                deduplicated_data.append(new_data)
                        else:
                            # 所有字段都不存在，跳过并警告
                            available_fields = [k for k in data.keys() if k not in ['source_field', 'original_content']]
                            print(f"警告: 记录缺少去重字段 {key_fields}，可用字段: {available_fields}")

                    except json.JSONDecodeError as e:
                        print(f"错误: JSON解析失败: {e}")
                        continue

        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时发生错误: {e}")
            continue

    print(f"去重统计: 从 {len(file_paths)} 个文件中提取 {len(deduplicated_data)} 条唯一记录")
    return deduplicated_data


def split_dataset(
        pos_files: List[str],
        neg_files: List[str],
        test_size: int = 100,
        key_fields: List[str] = ["text", "content"]  # 修改为列表
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    读取正负样本文件，划分数据集

    Args:
        pos_files: 正样本文件路径列表
        neg_files: 负样本文件路径列表
        test_size: 测试集大小（正负各多少条）
        key_fields: 用于去重的字段名列表（按优先级排序）

    Returns:
        (test_data, train_pos_data, train_neg_data)
    """
    # 读取并去重正负样本（使用新的多字段去重函数）
    print("正在读取正样本文件...")
    all_pos_data = read_and_deduplicate_jsonl_files(pos_files, key_fields)
    print(f"正样本去重后数量: {len(all_pos_data)}")

    print("正在读取负样本文件...")
    all_neg_data = read_and_deduplicate_jsonl_files(neg_files, key_fields)
    print(f"负样本去重后数量: {len(all_neg_data)}")

    # 检查样本数量是否足够
    if len(all_pos_data) < test_size:
        raise ValueError(f"正样本数量不足 {test_size} 条，当前只有 {len(all_pos_data)} 条")

    if len(all_neg_data) < test_size:
        raise ValueError(f"负样本数量不足 {test_size} 条，当前只有 {len(all_neg_data)} 条")

    # 随机打乱数据
    random.shuffle(all_pos_data)
    random.shuffle(all_neg_data)

    # 划分测试集
    test_pos = all_pos_data[:test_size]
    test_neg = all_neg_data[:test_size]

    # 训练集
    train_pos = all_pos_data[test_size:]
    train_neg = all_neg_data[test_size:]

    # 合并正负测试集
    test_data = test_pos + test_neg
    random.shuffle(test_data)  # 打乱测试集

    return test_data, train_pos, train_neg


def write_jsonl(data: List[Dict], file_path: str) -> None:
    """将数据写入JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # 命令行参数解析 - 修改为支持多个字段
    parser = argparse.ArgumentParser(description='划分正负样本数据集')
    parser.add_argument('--pos', nargs='+', required=True, help='正样本JSONL文件路径')
    parser.add_argument('--neg', nargs='+', required=True, help='负样本JSONL文件路径')
    parser.add_argument('--test-size', type=int, default=100, help='测试集正负样本数量（默认: 100）')
    parser.add_argument('--key-fields', nargs='+', default=['text', 'content'],
                        help='去重字段优先级列表（默认: text content）')
    parser.add_argument('--output-dir', default='./', help='输出目录（默认: 当前目录）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认: 42）')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print("=" * 50)
    print(f"正样本文件: {len(args.pos)} 个")
    print(f"负样本文件: {len(args.neg)} 个")
    print(f"测试集大小: 正负各 {args.test_size} 条")
    print(f"去重字段优先级: {args.key_fields}")
    print(f"随机种子: {args.seed}")
    print("=" * 50)

    try:
        # 划分数据集（使用新的多字段参数）
        test_data, train_pos, train_neg = split_dataset(
            pos_files=args.pos,
            neg_files=args.neg,
            test_size=args.test_size,
            key_fields=args.key_fields  # 传递字段列表
        )

        if not Path(args.output_dir).exists():
            print(f"创建output:{args.output_dir}")
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 定义输出文件路径
        test_file = f"{args.output_dir.rstrip('/')}/test.jsonl"
        train_pos_file = f"{args.output_dir.rstrip('/')}/train_pos.jsonl"
        train_neg_file = f"{args.output_dir.rstrip('/')}/train_neg.jsonl"

        # 写入文件
        print(f"正在写入测试集: {test_file}")
        write_jsonl(test_data, test_file)

        print(f"正在写入正样本训练集: {train_pos_file}")
        write_jsonl(train_pos, train_pos_file)

        print(f"正在写入负样本训练集: {train_neg_file}")
        write_jsonl(train_neg, train_neg_file)

        if len(train_pos) < len(train_neg):
            random.shuffle(train_neg)
            print(f"正在写入真正用于训练的1:1 负样本训练集， 条目数：{len(train_pos)}")
            write_jsonl(train_neg[:len(train_pos)], f"{args.output_dir.rstrip('/')}/train_neg_target.jsonl")

        # 打印统计信息
        print("\n" + "=" * 50)
        print("数据集划分完成！")
        print(f"测试集: {len(test_data)} 条 (正{args.test_size}+负{args.test_size})")
        print(f"正样本训练集: {len(train_pos)} 条")
        print(f"负样本训练集: {len(train_neg)} 条")
        print(f"总计: {len(test_data) + len(train_pos) + len(train_neg)} 条")
        print("=" * 50)

    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()