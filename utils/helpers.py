import pandas as pd
def load_best_parameters(filepath: str) -> dict:
    """Loads best parameters from a CSV file."""
    try:
        best_params = pd.read_csv(filepath).set_index("number").sort_values("value", ascending=False).iloc[0].to_dict()
        return best_params
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}
    

def parse_best_params(best_params: dict) -> tuple:
    """Parses the best parameters dictionary into individual variables."""
    TP, SL, min_signals = 0, 0, 0
    strategies = []
    for key, val in best_params.items():
        key = key.replace("params_", "")
        if key == "TP":
            TP = float(val)
        elif key == "SL":
            SL = float(val)
        elif key == "min_signals":
            min_signals = int(val)
        else:
            if val:
                strategies.append(key)
    return TP, SL, min_signals, strategies


def train_test_split(data: pd.DataFrame, ratio: float) -> tuple:
    """Splits data into training and test sets based on the ratio."""
    train_size = int(ratio * len(data))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    return train_data, test_data