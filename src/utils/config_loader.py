import yaml
from typing import Dict, Any


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    載入YAML配置檔案

    Args:
        config_path: 配置檔案的路徑

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
