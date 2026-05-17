import json
import os
import joblib
import pandas as pd
import numpy as np

def model_fn(model_dir):
    """Memuat model biner dari folder artifact."""
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan di {model_path}")
    return joblib.load(model_path)


def input_fn(request_body, request_content_type):
    """Menerima request JSON secara aman."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Mengeksekusi preprocessing pipeline dan prediksi akhir XGBoost."""
    data = input_data["instances"] if "instances" in input_data else input_data
    
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                          'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                          'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing']
    feature_names = categorical_features + numerical_features
    
    df = pd.DataFrame(data, columns=feature_names)
    
    pure_strings = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Age_group']
    for col in pure_strings:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    for col in ['CryoSleep', 'VIP']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: True if str(x) in ['1', '1.0', 'True', 'true', 'True '] else False)
    
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            
    df = df[feature_names]

    if hasattr(model, 'steps'):
        preprocessor = model.steps[0][1]
        classifier = model.steps[1][1]
        
        X_transformed = preprocessor.transform(df)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        X_transformed = np.nan_to_num(X_transformed)
        
        preds = classifier.predict(X_transformed)
    else:
        preds = model.predict(df)
        
    return {"status": "SUCCESS", "predictions": preds.tolist()}


def output_fn(prediction, response_content_type):
    """Mengembalikan respon prediksi berbentuk JSON."""
    return json.dumps(prediction), "application/json"