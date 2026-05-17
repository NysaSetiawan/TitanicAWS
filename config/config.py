from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_ING_DIR = PROJECT_ROOT / "data" / "ingested"

TARGET_COL = "Transported" 

DROP_COLS = ["PassengerId", "Name", "Cabin", "Group", "FirstName", "LastName"]

TEST_SIZE = 0.2
RANDOM_STATE = 42