import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_experiment_results(data: dict, file_path: str = r'result_data'):
    """
    将实验的所有结果（参数、耗时、精英策略等）保存为JSON文件。

    Args:
        data (dict): 包含所有要保存信息的字典。
        file_path (str, optional): 保存文件的路径.
    """

    # 2. 使用 Path 对象构建路径
    target_dir = Path(file_path)

    # 3. 创建目录
    target_dir.mkdir(parents=True, exist_ok=True)

    full_file_path = target_dir / f"experiment_results.json"

    try:
        with open(full_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log.info(f"实验结果已成功保存至: {full_file_path}")
        return full_file_path
    except Exception as e:
        log.error(f"保存结果时发生错误: {e}")
        return None
