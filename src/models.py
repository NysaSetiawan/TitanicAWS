from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from features.pipeline_preprocessor import build_preprocessor
RANDOM_STATE = 42

def build_pipelines(df=None) -> dict:
    """
    Membangun pipeline menggunakan list fitur manual Space Titanic yang diberikan.
    """
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                          'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                          'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing']

    print(f"\n[Space Titanic Manual List]")
    print(f"-> Total Fitur Numerik    : {len(numerical_features)} kolom")
    print(f"-> Total Fitur Kategorikal: {len(categorical_features)} kolom")

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    pipelines = {
        "LogisticRegression": Pipeline([
            ("preprocessing", preprocessor),
            ("clf", LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("preprocessing", preprocessor),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=6)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessing", preprocessor),
            ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", max_depth=5)),
        ])
    }
    return pipelines