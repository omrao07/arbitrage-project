import os
import json
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
MACRO_PATH = os.path.join(DATA_PATH, 'macro_data')

def load_macro_data(region: str) -> pd.DataFrame:
    """Loads macroeconomic data CSV for a specific region."""
    file_path = os.path.join(MACRO_PATH, f'{region.lower()}_macro.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Macro file not found: {file_path}")
    return pd.read_csv(file_path)

def load_economic_indicators() -> pd.DataFrame:
    """Loads the general economic indicators CSV."""
    file_path = os.path.join(DATA_PATH, 'economic_indicators.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError("economic_indicators.csv not found.")
    return pd.read_csv(file_path)

def load_fed_speeches() -> list:
    """Loads FED speeches JSON."""
    file_path = os.path.join(DATA_PATH, 'fed_speeches.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError("fed_speeches.json not found.")
    with open(file_path, 'r') as f:
        return json.load(f)

def load_recession_probabilities() -> dict:
    """Loads model-based recession probabilities (JSON)."""
    file_path = os.path.join(DATA_PATH, 'recession_probabilities.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError("recession_probabilities.json not found.")
    with open(file_path, 'r') as f:
        return json.load(f)

def load_config() -> dict:
    """Loads strategy toggle + weight config JSON."""
    file_path = os.path.join(DATA_PATH, 'config.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError("config.json not found.")
    with open(file_path, 'r') as f:
        return json.load(f)