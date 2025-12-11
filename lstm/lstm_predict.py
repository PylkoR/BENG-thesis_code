import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, 'lstm_output')
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_ret.csv')

PARAMS_PATH = os.path.join(ROOT_DIR, 'lstm_best_params.json')
FEATURES_PATH = os.path.join(ROOT_DIR, 'lstm_selected_features.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'training_results', 'best_lstm_model.keras')
SCALER_X_PATH = os.path.join(ROOT_DIR, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(ROOT_DIR, 'scaler_y.pkl')
PRED_DIR = os.path.join(ROOT_DIR, 'prediction_results')
os.makedirs(PRED_DIR, exist_ok=True)

# Pliki wynikowe
OUTPUT_PLOT_FULL = os.path.join(PRED_DIR, 'lstm_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(PRED_DIR, 'lstm_forecast_zoom.png')
OUTPUT_PRED_PATH = os.path.join(PRED_DIR, 'lstm_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(PRED_DIR, 'lstm_metrics.csv')

# Style wykresów
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

if __name__ == "__main__":
    print("--- PREDYKCJA LSTM (Full) ---")
    
    # 1. Ładowanie zasobów
    with open(PARAMS_PATH) as f: config = json.load(f)
    with open(FEATURES_PATH) as f: features = json.load(f)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    model = load_model(MODEL_PATH)
    
    SEQ = config['structure']['seq_len']
    SPLIT = config['meta']['split']
    TARGET = config['meta']['target']
    
    # 2. Dane
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    
    mask_train = df.index < SPLIT
    df_train = df.loc[mask_train]
    df_test = df.loc[~mask_train]
    
    # Buforowanie
    X_full = pd.concat([df_train[features].tail(SEQ), df_test[features]])
    y_full = pd.concat([df_train[[TARGET]].tail(SEQ), df_test[[TARGET]]])
    
    X_sc = scaler_X.transform(X_full)
    y_sc = scaler_y.transform(y_full)
    X_test, y_test = create_sequences(X_sc, y_sc, SEQ)
    
    # 3. Predykcja
    pred_sc = model.predict(X_test, verbose=0)
        
    pred_ret = scaler_y.inverse_transform(pred_sc).flatten()
    act_ret = scaler_y.inverse_transform(y_test).flatten()
    
    # 4. Rekonstrukcja Cen
    actual_prices = df_test['mwig40_Zamkniecie'].values
    last_train_price = df_train['mwig40_Zamkniecie'].iloc[-1]
    
    # Wyrównanie
    min_len = min(len(actual_prices), len(pred_ret))
    actual_prices = actual_prices[:min_len]
    pred_ret = pred_ret[:min_len]
    act_ret = act_ret[:min_len]
    dates = df_test.index[:min_len]
    
    # One-Step-Ahead: Cena(t) = Prawdziwa(t-1) * exp(Zwrot(t))
    base_prices = np.concatenate(([last_train_price], actual_prices[:-1]))
    pred_prices = base_prices * np.exp(pred_ret)
    
    # 5. Metryki
    mae = mean_absolute_error(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    dir_acc = np.mean(np.sign(pred_ret) == np.sign(act_ret)) * 100
    
    print(f"\n--- WYNIKI MODELU LSTM ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.4f} %")
    print(f"Dir Accuracy: {dir_acc:.2f} %")
    
    # 6. Zapis Metryk do CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy_Pct'],
        'Value': [mae, rmse, mape, dir_acc]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)
    print(f"Zapisano metryki: {OUTPUT_METRICS_PATH}")

    # 7. Eksport wyników predykcji
    results_df = pd.DataFrame({
        'Data': dates,
        'Rzeczywiste': actual_prices,
        'Prognoza_Cena': pred_prices,
        'Prognoza_Zwrot': pred_ret,
        'Rzeczywisty_Zwrot': act_ret
    })
    results_df.set_index('Data', inplace=True)
    results_df.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',')
    print(f"Zapisano wyniki predykcji: {OUTPUT_PRED_PATH}")

    # 8. Wizualizacja
    
    # Wykres 1: Pełny
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index[-200:], df_train['mwig40_Zamkniecie'].tail(200), label='Dane Treningowe')
    plt.plot(results_df.index, results_df['Rzeczywiste'], label='Dane Rzeczywiste', color='green')
    plt.plot(results_df.index, results_df['Prognoza_Cena'], label='Prognoza LSTM', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Prognoza LSTM vs Rzeczywistość', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL, dpi=200)
    print(f"Zapisano wykres główny: {OUTPUT_PLOT_FULL}")

    # Wykres 2: Zoom
    start_date_zoom = '2025-01-02'
    if start_date_zoom in results_df.index or results_df.index[-1] > pd.to_datetime(start_date_zoom):
        try:
            plot_data = results_df.loc[start_date_zoom:]
            plt.figure(figsize=(12, 6))
            plt.plot(plot_data.index, plot_data['Rzeczywiste'], label='Dane Rzeczywiste', linewidth=2, color='green')
            plt.plot(plot_data.index, plot_data['Prognoza_Cena'], label='Prognoza LSTM', color='red', linestyle='--', linewidth=2)
            
            plt.title(f'Szczegóły prognozy od {start_date_zoom}', fontsize=TITLE_FONT_SIZE)
            plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
            plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
            plt.legend(fontsize=TICK_FONT_SIZE)
            plt.grid(True, alpha=0.3)
            plt.savefig(OUTPUT_PLOT_ZOOM, dpi=200)
            print(f"Zapisano wykres zoom: {OUTPUT_PLOT_ZOOM}")
        except Exception:
            print("Brak danych do wykresu zoom.")
            
    print("Zakończono.")