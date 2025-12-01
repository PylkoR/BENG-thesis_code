import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Konfiguracja ścieżek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset.csv')

# Ścieżki wyjściowe
OUTPUT_PLOT_FULL = os.path.join(SCRIPT_DIR, 'naive_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(SCRIPT_DIR, 'naive_forecast_zoom.png')
OUTPUT_PRED_PATH = os.path.join(SCRIPT_DIR, 'naive_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(SCRIPT_DIR, 'naive_metrics.csv')

# Konfiguracja wykresów
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12

try:
    # 1. Wczytanie i przygotowanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.set_index('Data').sort_index()

    target_col = 'mwig40_Zamkniecie'
    
    # 2. Implementacja Metody Naiwnej
    df['Naive_Forecast'] = df[target_col].shift(1)
    
    # Usuwamy pierwszy wiersz (NaN po przesunięciu)
    df = df.dropna()

    data = df[[target_col, 'Naive_Forecast']].copy()

    # 3. Podział na zbiór treningowy i testowy (90/10)
    train_size = int(len(data) * 0.9)
    train, test = data.iloc[:train_size], data.iloc[train_size:].copy()

    print(f"Zbiór treningowy: {len(train)} obserwacji")
    print(f"Zbiór testowy: {len(test)} obserwacji")

    # 4. Obliczenie metryk błędu
    mae = mean_absolute_error(test[target_col], test['Naive_Forecast'])
    rmse = np.sqrt(mean_squared_error(test[target_col], test['Naive_Forecast']))
    mape = np.mean(np.abs((test[target_col] - test['Naive_Forecast']) / test[target_col])) * 100

    print(f"\nWyniki metody naiwnej:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Zapis metryk do pliku CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE'],
        'Value': [mae, rmse, mape]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)
    print(f"Zapisano metryki do: {OUTPUT_METRICS_PATH}")

    # 5. Wizualizacja wyników

    # Wykres 1: Szerszy kontekst (ostatnie 200 dni treningu + test)
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-200:], train[target_col].iloc[-200:], label='Dane Treningowe')
    plt.plot(test.index, test[target_col], label='Dane Rzeczywiste')
    plt.plot(test.index, test['Naive_Forecast'], label='Prognoza Naiwna', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Prognoza Naiwna vs Rzeczywistość', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.xticks(fontsize = TICK_FONT_SIZE)
    plt.yticks(fontsize = TICK_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL, dpi=200)
    print(f"Zapisano wykres ogólny do: {OUTPUT_PLOT_FULL}")

    # Wykres 2: Zoom na ostatnie 100 dni prognozy
    plt.figure(figsize=(12, 6))
    last_100_days = test.index[-100:]
    plt.plot(last_100_days, test[target_col].tail(100), label='Dane Rzeczywiste', linewidth=2)
    plt.plot(last_100_days, test['Naive_Forecast'].tail(100), label='Prognoza Naiwna', color='red', linestyle='--', linewidth=2)

    plt.title('Szczegóły prognozy (ostatnie 100 dni)', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.xticks(fontsize = TICK_FONT_SIZE)
    plt.yticks(fontsize = TICK_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM, dpi=200)
    print(f"Zapisano wykres szczegółowy do: {OUTPUT_PLOT_ZOOM}")

    plt.show()

    # 6. Zapis wyników predykcji
    test.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',')
    print(f"Zapisano prognozy do: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Wystąpił błąd krytyczny: {e}")