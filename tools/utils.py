import toml
from string import Template


def load_config_with_references(config_path):
    # 加载原始TOML配置
    with open(config_path, "r") as f:
        config = toml.load(f)

    # 将配置转换为字符串进行模板替换
    config_str = toml.dumps(config)
    template = Template(config_str)

    # 执行替换（基于已加载的配置值）
    resolved_config_str = template.safe_substitute(**flatten_config(config))

    # 重新解析为字典
    return toml.loads(resolved_config_str)


def flatten_config(config, parent_key='', sep='.'):
    """将嵌套配置展平为单层字典"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == '__main__':
# 使用示例
    config = load_config_with_references("config.toml")
    print(config["data"]["positive_samples"])  # 输出: ./data/positive.jsonl