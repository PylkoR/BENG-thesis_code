import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# --- KONFIGURACJA ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, 'lstm_output')
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_ret.csv')

PARAMS_PATH = os.path.join(ROOT_DIR, 'lstm_best_params.json')
FEATURES_PATH = os.path.join(ROOT_DIR, 'lstm_selected_features.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'training_results', 'best_lstm_model.keras')
SCALER_X_PATH = os.path.join(ROOT_DIR, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(ROOT_DIR, 'scaler_y.pkl')

OUTPUT_PLOT = os.path.join(ROOT_DIR, 'feature_importance.png')
OUTPUT_CSV = os.path.join(ROOT_DIR, 'feature_importance.csv')

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

if __name__ == "__main__":
    print("--- Globalna analiza wrażliwości ---")
    
    # 1. Wczytywanie zasobów
    with open(PARAMS_PATH) as f: config = json.load(f)
    with open(FEATURES_PATH) as f: features = json.load(f)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    model = load_model(MODEL_PATH)
    
    SEQ = config['structure']['seq_len']
    SPLIT = config['meta']['split']
    TARGET = config['meta']['target']
    
    # 2. Przygotowanie danych testowych
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    
    mask_test = df.index >= SPLIT

    df_train_tail = df.loc[df.index < SPLIT].tail(SEQ)
    df_test_full = pd.concat([df_train_tail, df.loc[mask_test]])
    
    X_raw = df_test_full[features]
    y_raw = df_test_full[[TARGET]]
    
    # Transformacja
    X_sc = scaler_X.transform(X_raw)
    y_sc = scaler_y.transform(y_raw)
    
    X_test, y_test = create_sequences(X_sc, y_sc, SEQ)
    
    print(f"Liczba próbek testowych: {len(X_test)}")
    
    # 3. Obliczenie błędu bazowego (Baseline)
    print("Liczenie błędu bazowego...")
    pred_base = model.predict(X_test, verbose=0)
    baseline_mse = mean_squared_error(y_test, pred_base)
    print(f"Baseline MSE: {baseline_mse:.8f}")
    
    # 4. Pętla permutacji (Tasowanie każdej cechy)
    importances = {}
    
    for i, feat_name in enumerate(features):
        print(f"Analiza cechy: {feat_name}...")
        
        # Kopia danych, żeby nie psuć oryginału
        X_shuffled = X_test.copy()
        
        np.random.shuffle(X_shuffled[:, :, i])
        
        # Predykcja na popsutych danych
        pred_shuffled = model.predict(X_shuffled, verbose=0)
        shuffled_mse = mean_squared_error(y_test, pred_shuffled)
        
        # Ważność = O ile wzrósł błąd?
        importance = shuffled_mse - baseline_mse
        importances[feat_name] = importance
        
    # 5. Wyniki i Wizualizacja
    results = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance_MSE_Drop'])
    results = results.sort_values(by='Importance_MSE_Drop', ascending=True) 
    
    print("\n--- WYNIKI WRAŻLIWOŚCI ---")
    print(results.sort_values(by='Importance_MSE_Drop', ascending=False))
    
    results.to_csv(OUTPUT_CSV, sep=';', decimal=',', index=False)
    
    # Wykres
    plt.figure(figsize=(12, 8))
    # Kolory: Niebieski (Ważna), Czerwony (Szkodliwa - zmniejsza błąd po usunięciu)
    colors = ['red' if x <= 0 else 'skyblue' for x in results['Importance_MSE_Drop']]
    
    plt.barh(results['Feature'], results['Importance_MSE_Drop'], color=colors)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.xlabel('Wzrost błędu MSE po przetasowaniu (Więcej = Ważniejsza cecha)')
    plt.title(f'Ważność Cech LSTM (Permutation Importance)\nBaseline MSE: {baseline_mse:.2e}')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\nWykres zapisano: {OUTPUT_PLOT}")