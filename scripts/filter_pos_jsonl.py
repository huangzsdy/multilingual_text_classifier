import json
import sys
from pathlib import Path
from tqdm import tqdm


def filter_jsonl_by_pos_prob(input_path, output_path, threshold):
    """
    过滤JSONL文件，只保留model_pred.pos_prob大于阈值的条目。

    参数:
        input_path (str): 输入JSONL文件的路径。
        output_path (str): 输出JSONL文件的路径。
        threshold (float): 阈值，用于比较pos_prob。
    """
    filtered_entries = []

    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f_in:
        total_lines = sum(1 for line in f_in)
        f_in.seek(0)

        for line in tqdm(f_in, total=total_lines, desc="读取数据", unit="sample"):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                entry = json.loads(line)  # 解析JSON对象
                # 检查model_pred.pos_prob是否存在且大于阈值
                if 'prediction' in entry:
                    if entry['prediction']['probabilities']:
                        pos_prob = entry['prediction']['probabilities']['positive']
                    elif entry['prediction']['pos_prob']:
                        pos_prob = entry['prediction']['pos_prob']
                    assert pos_prob > 0.0 , f"pos_prob:{pos_prob} 必须大于0"
                    if pos_prob > threshold:
                        filtered_entries.append(entry)
                elif 'model_pred' in entry:
                    if entry['model_pred']['probabilities']:
                        pos_prob = entry['model_pred']['probabilities']['positive']
                    elif entry['model_pred']['pos_prob']:
                        pos_prob = entry['model_pred']['pos_prob']
                    assert pos_prob > 0.0 , f"pos_prob:{pos_prob} 必须大于0"
                    if pos_prob > threshold:
                        filtered_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"警告：跳过无效的JSON行: {line}，错误: {e}")
            except Exception as err:
                print(f"处理失败:{line}, error:{err}")




    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in tqdm(filtered_entries, total=len(filtered_entries), desc="写入数据", unit="sample"):
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"过滤完成！共处理{len(filtered_entries)}条符合条件的条目，已写入: {output_path}")


# 示例使用（需根据实际路径和阈值修改）
if __name__ == "__main__":
    # input_file = "text_zh_pred_json.txt"  # 替换为您的输入文件路径
    # output_file = "filtered_output.jsonl"  # 替换为您的输出文件路径
    input_path, output_path, threshold_value = sys.argv[1:]
    threshold_value = float(threshold_value)
    # threshold_value = 0.5  # 替换为您的阈值，例如0.5

    input_p = Path(input_path)
    output_p = Path(output_path)

    # 如果输入是目录，遍历处理目录下所有jsonl文件
    if input_p.is_dir():
        output_p.mkdir(parents=True, exist_ok=True)
        for jsonl_file in input_p.glob("*.jsonl"):
            output_file = output_p / jsonl_file.name
            print(f"处理文件: {jsonl_file} -> {output_file}")
            filter_jsonl_by_pos_prob(str(jsonl_file), str(output_file), threshold_value)
    else:
        # 单文件处理
        filter_jsonl_by_pos_prob(input_path, output_path, threshold_value)