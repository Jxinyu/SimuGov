import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_experiment_results(data: dict, file_path: str = r'result_data'):
    """
    Save all results of the experiment (parameters, time consumption, elite strategies, etc.) as a JSON file.

    Args:
        data (dict): A dictionary containing all information to be saved.
        file_path (str, optional): The path to save the file.
    """

    # 2. Use Path object to build path
    target_dir = Path(file_path)

    # 3. Create directory
    target_dir.mkdir(parents=True, exist_ok=True)

    full_file_path = target_dir / f"experiment_results.json"

    try:
        with open(full_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log.info(f"Experiment results successfully saved to: {full_file_path}")
        return full_file_path
    except Exception as e:
        log.error(f"Error occurred while saving results: {e}")
        return None
