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
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.csv')

# Katalogi z artefaktami (model, params, skalery)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'training_output')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'best_params.json')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'lstm_model.keras')
SCALER_FEAT_PATH = os.path.join(OUTPUT_DIR, 'scaler_features.pkl')
SCALER_TARG_PATH = os.path.join(OUTPUT_DIR, 'scaler_target.pkl')

# Katalog na wyniki
PRED_DIR = os.path.join(SCRIPT_DIR, 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

# Ścieżki wyjściowe
OUTPUT_PLOT_FULL = os.path.join(PRED_DIR, 'lstm_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(PRED_DIR, 'lstm_forecast_zoom.png')
OUTPUT_PRED_PATH = os.path.join(PRED_DIR, 'lstm_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(PRED_DIR, 'lstm_metrics.csv')

# Styl
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TEST_SPLIT = 0.1

try:
    # 1. Wczytanie konfiguracji
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError("Brak pliku best_params.json")

    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    
    SEQ_LEN = params['seq_len']
    TARGET_COL = params.get('target_col', 'mwig40_Zamkniecie')
    print(f"Konfiguracja: SEQ_LEN={SEQ_LEN}, Cel={TARGET_COL}")

    # 2. Wczytanie modelu i skalerów
    print("Ładowanie modelu i skalerów...")
    model = load_model(MODEL_PATH)
    scaler_features = joblib.load(SCALER_FEAT_PATH)
    scaler_target = joblib.load(SCALER_TARG_PATH)

    # 3. Wczytanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()

    feature_cols = df.columns.tolist()

    # 4. Podział na zbiór testowy
    train_size = int(len(df) * (1 - TEST_SPLIT))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Zbiór testowy: {len(test_df)} dni")

    # 5. Przygotowanie danych wejściowych (X)
    input_data = pd.concat((train_df.iloc[-SEQ_LEN:], test_df))
    
    # Skalowanie
    scaled_input = scaler_features.transform(input_data[feature_cols])

    X_test = []
    for i in range(len(test_df)):
        X_test.append(scaled_input[i : i + SEQ_LEN])
    X_test = np.array(X_test)

    # 6. Predykcja zwrotów
    print("Generowanie prognoz (Log Returns)...")
    pred_returns_scaled = model.predict(X_test)
    
    # Odwracanie skalowania
    pred_returns = scaler_target.inverse_transform(pred_returns_scaled).flatten()

    # 7. Rekonstrukcja ceny
    # Cena_t = Cena_{t-1} * exp(Zwrot_t)
    print("Rekonstrukcja cen z prognozowanych zwrotów...")
    
    # Pobieranie prawdziwych cen
    if 'mwig40_Zamkniecie' not in test_df.columns:
        raise ValueError("Brak kolumny 'mwig40_Zamkniecie' do weryfikacji ceny!")

    actual_prices = test_df['mwig40_Zamkniecie'].values
    
    # Potrzebujemy ceny z dnia POPRZEDZAJĄCEGO każdą predykcję
    # Dla pierwszego dnia testu, jest to ostatnia cena treningu
    last_train_price = train_df['mwig40_Zamkniecie'].iloc[-1]
    
    # Tworzymy wektor cen "wczorajszych" (P_{t-1})
    # [Ostatni_Train, Test_0, Test_1, ... Test_N-1]
    prev_prices = np.concatenate(([last_train_price], actual_prices[:-1]))
    
    # Obliczamy prognozowaną cenę
    predicted_prices = prev_prices * np.exp(pred_returns)
    
    # Konwersja na Pandas Series
    forecast_series = pd.Series(predicted_prices, index=test_df.index)
    actual_series = test_df['mwig40_Zamkniecie']

    # 8. Metryki
    mae = mean_absolute_error(actual_series, forecast_series)
    rmse = np.sqrt(mean_squared_error(actual_series, forecast_series))
    mape = np.mean(np.abs((actual_series - forecast_series) / actual_series)) * 100
    
    # Dodatkowa metryka
    actual_returns = test_df['mWIG40_Ret'].values
    # Sprawdzamy czy znak prognozy zgadza się ze znakiem rzeczywistości
    direction_match = np.sign(pred_returns) == np.sign(actual_returns)
    dir_acc = np.mean(direction_match) * 100

    print(f"\n--- WYNIKI MODELU LSTM (CENA) ---")
    print(f"MAE:  {mae:.2f} pkt")
    print(f"RMSE: {rmse:.2f} pkt")
    print(f"MAPE: {mape:.2f} %")
    print(f"Trafność kierunku (Dir Accuracy): {dir_acc:.2f} %")

    # Zapis metryk
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy'],
        'Value': [mae, rmse, mape, dir_acc]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)

    # 9. Wizualizacja
    
    # Wykres 1: Full
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index[-200:], train_df['mwig40_Zamkniecie'].tail(200), label='Historia')
    plt.plot(test_df.index, actual_series, label='Rzeczywistość')
    plt.plot(test_df.index, forecast_series, label='Prognoza LSTM', color='red', linestyle='--', alpha=0.8)
    
    plt.title('LSTM: Prognoza Ceny (na bazie zwrotów)', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL)
    print(f"Zapisano wykres: {OUTPUT_PLOT_FULL}")

    # Wykres 2: Zoom
    plt.figure(figsize=(12, 6))
    last_100 = test_df.index[-100:]
    plt.plot(last_100, actual_series.tail(100), label='Rzeczywistość', linewidth=2)
    plt.plot(last_100, forecast_series.tail(100), label='Prognoza LSTM', color='red', linestyle='--', linewidth=2)
    
    plt.title('Szczegóły prognozy (ostatnie 100 dni)', fontsize=TITLE_FONT_SIZE)
    plt.ylabel('mWIG40')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM)
    print(f"Zapisano zoom: {OUTPUT_PLOT_ZOOM}")
    
    plt.show()

    # Zapis wyników
    results_df = pd.DataFrame({
        'Rzeczywiste': actual_series,
        'Prognoza_Cena': forecast_series,
        'Prognoza_Zwrot': pred_returns,
        'Rzeczywisty_Zwrot': test_df['mWIG40_Ret']
    })
    results_df.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',')
    print(f"Dane zapisano do: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")