import pandas as pd
from sklearn.model_selection import train_test_split

# Mengambil variabel dari config kamu
from config.config import (
    DATA_RAW_DIR,
    DATA_ING_DIR,
    TARGET_COL,
    DROP_COLS,
    RANDOM_STATE,
    TEST_SIZE,
)

# Target biner untuk Space Titanic
CLASS_NAMES = ["not_transported", "transported"]

def ingest_data() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = DATA_RAW_DIR / "train.csv"
    ing_file = DATA_ING_DIR / "train.csv"

    df = pd.read_csv(raw_file)
    assert not df.empty, "Dataset is empty"

    df.to_csv(ing_file, index=False)
    print(f"Data ingested: {raw_file} → {ing_file}")


def load_dataset() -> pd.DataFrame:
    path = DATA_ING_DIR / "train.csv"
    if not path.exists():
        ingest_data()
        
    df = pd.read_csv(path)
    # 1. Extract Cabin features
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
    df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')
    # 2. Extract group and individual from PassengerId
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')
    df['Solo'] = (df['Group_size'] == 1).astype(int)
    # 3. Extract first and last name
    df['FirstName'] = df['Name'].apply(lambda x: x.split()[0] if pd.notna(x) else 'Unknown')
    df['LastName'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
    df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')
    # 4. Total spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    # 5. Spending ratios (Tidak masuk list manual model, tapi biarkan tetap terhitung)
    for col in spending_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)
    # 6. Age groups
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], 
                             labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
    df['Age_group'] = df['Age_group'].astype(str)
    # 7. Missing value indicators
    df['Age_missing'] = df['Age'].isna().astype(int)
    df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)

    if TARGET_COL in df.columns:
        df = df.rename(columns={TARGET_COL: "target"})
        
    return df


def split_data(df: pd.DataFrame):
    extra_drops = ["target", "Group", "FirstName", "LastName"] + DROP_COLS
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                          'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                          'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing']
    baku_features = categorical_features + numerical_features

    X = df[baku_features]
    y = df["target"]
    
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )