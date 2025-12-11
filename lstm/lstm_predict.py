import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- KONFIGURACJA ŚCIEŻEK ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_ROOT = os.path.join(SCRIPT_DIR, 'lstm_output')
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_lstm.csv')

# Wejście (z Tunera i Treningu)
PARAMS_PATH = os.path.join(LSTM_ROOT, 'lstm_best_params.json')
SCALER_X_PATH = os.path.join(LSTM_ROOT, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(LSTM_ROOT, 'scaler_y.pkl')
MODEL_PATH = os.path.join(LSTM_ROOT, 'training_results', 'best_lstm_model.keras')

# Wyjście (Predykcja)
PREDICT_DIR = os.path.join(LSTM_ROOT, 'prediction_results')
os.makedirs(PREDICT_DIR, exist_ok=True)

# --- FUNKCJE ---
def create_sequences(X_data, y_data, seq_len):
    X, y = [], []
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i:(i + seq_len)])
        y.append(y_data[i + seq_len])
    return np.array(X), np.array(y)

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Ładowanie zasobów
        print("Loading resources...")
        with open(PARAMS_PATH, 'r') as f:
            config = json.load(f)
            
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        model = load_model(MODEL_PATH)
        
        SEQ_LEN = config['optimal_structure']['seq_len']
        SPLIT_DATE = config['meta']['split_date']
        TARGET_COL = config['meta']['target_col']

        # 2. Przygotowanie danych TESTOWYCH
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()
        
        feature_cols = [c for c in df.columns if c != TARGET_COL]
        
        # Podział
        mask_train = df.index < SPLIT_DATE
        df_train = df.loc[mask_train]
        df_test = df.loc[~mask_train]
        
        # Budowa bufora dla predykcji (ostatnie SEQ_LEN z treningu + test)
        full_X = pd.concat([df_train[feature_cols].tail(SEQ_LEN), df_test[feature_cols]])
        full_y = pd.concat([df_train[[TARGET_COL]].tail(SEQ_LEN), df_test[[TARGET_COL]]])
        
        # Transformacja (TYLKO transform, bez fit)
        X_test_scaled = scaler_X.transform(full_X)
        y_test_scaled = scaler_y.transform(full_y)
        
        X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, SEQ_LEN)
        print(f"Prediction set shape: {X_test.shape}")

        # 3. Generowanie prognoz (Zwroty)
        pred_scaled = model.predict(X_test, verbose=0)
        
        pred_returns = scaler_y.inverse_transform(pred_scaled).flatten()
        actual_returns = scaler_y.inverse_transform(y_test).flatten()
        
        # Indeks czasowy (powinien idealnie pasować do df_test)
        pred_index = df_test.index

        # 4. Rekonstrukcja Cen (ONE-STEP-AHEAD)
        # Cena(t) = Prawdziwa_Cena(t-1) * exp(Zwrot(t))
        
        # Pobieramy prawdziwe ceny zamknięcia z okresu testowego
        # Musimy mieć też cenę z dnia PRZED pierwszym dniem testu
        actual_prices_test = df_test['mwig40_Zamkniecie'].values
        last_train_price = df_train['mwig40_Zamkniecie'].iloc[-1]
        
        # Tworzymy wektor cen bazowych: [Ostatnia_Train, ...Test_do_przedostatniego]
        base_prices = np.concatenate(([last_train_price], actual_prices_test[:-1]))
        
        # Rekonstrukcja: Bierzemy PRAWDZIWĄ cenę z wczoraj i mnożymy przez prognozowany zwrot
        predicted_prices_one_step = base_prices * np.exp(pred_returns)
            
        forecast_series = pd.Series(predicted_prices_one_step, index=pred_index)
        actual_series = df_test['mwig40_Zamkniecie']

        # 5. Metryki
        mae = mean_absolute_error(actual_series, forecast_series)
        rmse = np.sqrt(mean_squared_error(actual_series, forecast_series))
        mape = np.mean(np.abs((actual_series - forecast_series) / actual_series)) * 100
        
        # Direction Accuracy
        dir_acc = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100
        
        print(f"\n--- LSTM RESULTS ---")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"Dir Accuracy: {dir_acc:.2f}%")
        
        # Zapis metryk
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy_Pct'],
            'Value': [mae, rmse, mape, dir_acc]
        })
        metrics_path = os.path.join(PREDICT_DIR, 'lstm_metrics.csv')
        metrics_df.to_csv(metrics_path, sep=';', decimal=',', index=False)

        # 6. Eksport danych
        results_df = pd.DataFrame({
            'Date': pred_index,
            'Actual_Price': actual_series.values,
            'Predicted_Price': forecast_series.values,
            'Actual_Return': actual_returns,
            'Predicted_Return': pred_returns
        })
        results_path = os.path.join(PREDICT_DIR, 'lstm_predictions.csv')
        results_df.to_csv(results_path, sep=';', decimal=',', index=False)

        # 7. Wykresy
        # Full Chart
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index[-150:], df_train['mwig40_Zamkniecie'].tail(150), label='Train (Last 150)')
        plt.plot(actual_series.index, actual_series, label='Actual Test', color='green')
        plt.plot(forecast_series.index, forecast_series, label='LSTM Forecast', color='red', linestyle='--')
        plt.title('LSTM Forecast vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PREDICT_DIR, 'lstm_forecast_full.png'))
        
        # Zoom Chart
        zoom_start = '2025-01-01'
        if zoom_start in actual_series.index:
            plt.figure(figsize=(12, 6))
            plt.plot(actual_series.loc[zoom_start:], label='Actual', color='green')
            plt.plot(forecast_series.loc[zoom_start:], label='Forecast', color='red', linestyle='--')
            plt.title(f'LSTM Zoom (from {zoom_start})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(PREDICT_DIR, 'lstm_forecast_zoom.png'))
            
        print(f"\nResults saved to: {PREDICT_DIR}")

    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()