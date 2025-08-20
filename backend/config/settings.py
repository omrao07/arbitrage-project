# settings.py
"""
Global settings and configuration loader for the multi-region arbitrage & hedge fund platform.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# --- Load Environment Variables ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"[WARN] .env file not found at {ENV_PATH}")

# --- Application Settings ---
APP_NAME = "Global Arbitrage & Hedge Fund Platform"
APP_ENV = os.getenv("APP_ENV", "development")  # development / production
DEBUG = APP_ENV == "development"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Redis Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# --- API Keys ---
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
Zerodha_API_KEY = os.getenv("ZERODHA_API_KEY", "")
Zerodha_SECRET_KEY = os.getenv("ZERODHA_SECRET_KEY", "")
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
IBKR_API_KEY = os.getenv("IBKR_API_KEY", "")

# --- Data & Config Paths ---
CONFIG_DIR = BASE_DIR / "config"
FEEDS_DIR = CONFIG_DIR / "feeds"
REGISTER_FILE = CONFIG_DIR / "register.yaml"

# --- Load Register YAML ---
def load_register_config():
    """Load the register.yaml file for all feeds."""
    if not REGISTER_FILE.exists():
        raise FileNotFoundError(f"Register file not found: {REGISTER_FILE}")
    with open(REGISTER_FILE, "r") as f:
        return yaml.safe_load(f)

REGISTER_CONFIG = load_register_config()

# --- Risk Management Defaults ---
MAX_DRAWDOWN_PCT = 5
STOP_LOSS_PCT = 2
MAX_OPEN_POSITIONS = 20

# --- Strategy Defaults ---
DEFAULT_ALPHA_STRATEGIES = [
    "mean_reversion",
    "breakout",
    "pairs_trading"
]
DEFAULT_DIVERSIFIED_STRATEGIES = [
    "macro_rotation",
    "sector_rotation"
]

# --- Utility ---
def get_env_var(name: str, default: str = "") -> str:
    """Get environment variable or default."""
    return os.getenv(name, default)

if __name__ == "__main__":
    print(f"[INFO] Loaded settings for {APP_NAME} in {APP_ENV} mode")
    print(f"[INFO] Loaded feeds from {REGISTER_FILE}:")
    for feed in REGISTER_CONFIG.get("feeds", []):
        print(f" - {feed['region']}: {feed['config_file']}")