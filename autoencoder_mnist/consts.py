import logging
from pathlib import Path

# LOGGER
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

# DIRECTORIES
PROJECT_DIR = Path(__file__).parents[1]

# # DATA DIRECTORIES
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# # MODEL DIRECTORIES
MODEL_DIR = PROJECT_DIR / "models"
