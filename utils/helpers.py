import pandas as pd
import warnings
import logging
import os

def initialize_logging(log_dir: str) -> None:
    warnings.filterwarnings('ignore')

    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "working.log")

    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Logging Initialized")

def train_test_split(data: pd.DataFrame, ratio: float) -> tuple:
    """Splits data into training and test sets based on the ratio."""
    train_size = int(ratio * len(data))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    return train_data, test_data