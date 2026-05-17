from pathlib import Path

# 1. Konfigurasi Direktori/Path Folder
# Membantu Python melacak posisi folder data dari mana saja skrip dijalankan
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_ING_DIR = PROJECT_ROOT / "data" / "ingested"

# 2. Konfigurasi Kolom Dataset Space Titanic
TARGET_COL = "Transported"  # Kolom target prediksi (True/False atau 1/0)

# Kolom mentah bawaan yang tidak digunakan langsung dalam pemodelan 
# karena sudah dipecah/diolah di tahap Feature Engineering kamu
DROP_COLS = ["PassengerId", "Name", "Cabin", "Group", "FirstName", "LastName"]

# 3. Hyperparameter untuk Splitting Data
TEST_SIZE = 0.2
RANDOM_STATE = 42