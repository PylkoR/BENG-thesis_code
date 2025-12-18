import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Pomiar czasu - start
start_time = time.time()

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

# Data podziału na treningowy i testowy
SPLIT_DATE = '2022-01-03'

try:
    # 1. Wczytanie i przygotowanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.set_index('Data').sort_index()

    target_col = 'mwig40_Zamkniecie'
    
    # 2. Cena przesunięta o jeden dzień jako prognoza
    df['Naive_Forecast'] = df[target_col].shift(1)
    
    # Usuwamy pierwszy wiersz (NaN po przesunięciu)
    df = df.dropna()

    data = df[[target_col, 'Naive_Forecast']].copy()

    # 3. Podział na zbiór treningowy i testowy od daty SPLIT_DATE
    mask_train = data.index < SPLIT_DATE
    mask_test = data.index >= SPLIT_DATE

    train = data.loc[mask_train]
    test = data.loc[mask_test]

    print(f"Zbiór treningowy: {len(train)} obserwacji")
    print(f"Zbiór testowy: {len(test)} obserwacji")

    # 4. Obliczenie Directional Accuracy
    results_df = test.copy()
    
    # Ostatnia wartość treningowa dla ciągłości
    last_train_val = train[target_col].iloc[-1]
    
    # Rzeczywisty zwrot
    results_df['Prev_Actual'] = results_df[target_col].shift(1)
    results_df.loc[results_df.index[0], 'Prev_Actual'] = last_train_val
    results_df['Rzeczywisty_Zwrot'] = results_df[target_col] - results_df['Prev_Actual']

    # Prognozowany zwrot (różnica między prognozami)
    results_df['Prev_Forecast'] = results_df['Naive_Forecast'].shift(1)
    
    if len(train) >= 2:
        results_df.loc[results_df.index[0], 'Prev_Forecast'] = train[target_col].iloc[-2]
    else:
        results_df.loc[results_df.index[0], 'Prev_Forecast'] = last_train_val

    results_df['Prognozowany_Zwrot'] = results_df['Naive_Forecast'] - results_df['Prev_Forecast']

    direction_match = np.sign(results_df['Rzeczywisty_Zwrot']) == np.sign(results_df['Prognozowany_Zwrot'])
    dir_acc = np.mean(direction_match) * 100

    # 5. Obliczenie metryk błędu
    mae = mean_absolute_error(test[target_col], test['Naive_Forecast'])
    rmse = np.sqrt(mean_squared_error(test[target_col], test['Naive_Forecast']))
    mape = np.mean(np.abs((test[target_col] - test['Naive_Forecast']) / test[target_col])) * 100

    # Pomiar czasu - koniec
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nWyniki metody naiwnej:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"Direction Accuracy: {dir_acc:.2f}%")
    print(f"Czas wykonania: {execution_time:.4f} s")

    # Zapis metryk do pliku CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy_Pct', 'Execution_Time_Sec'],
        'Value': [mae, rmse, mape, dir_acc, execution_time]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)
    print(f"Zapisano metryki do: {OUTPUT_METRICS_PATH}")

    # 6. Wizualizacja wyników

    # Wykres 1: Szerszy kontekst
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-200:], train[target_col].iloc[-200:], label='Dane Treningowe')
    plt.plot(test.index, test[target_col], label='Dane Rzeczywiste', color='green')
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
    start_date = '2025-01-02'
    plot_data = test.loc[start_date:]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_data.index, plot_data[target_col], label='Dane Rzeczywiste', linewidth=2, color='green')
    plt.plot(plot_data.index, plot_data['Naive_Forecast'], label='Prognoza Naiwna', color='red', linestyle='--', linewidth=2)

    plt.title('Szczegóły prognozy od 2025 roku', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.xticks(fontsize = TICK_FONT_SIZE)
    plt.yticks(fontsize = TICK_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM, dpi=200)
    print(f"Zapisano wykres szczegółowy do: {OUTPUT_PLOT_ZOOM}")

    plt.show()

    # 7. Zapis wyników predykcji
    results_df_export = results_df.drop(columns=['Prev_Actual', 'Prev_Forecast'])
    results_df_export.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',')
    print(f"Zapisano prognozy i zwroty do: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Wystąpił błąd krytyczny: {e}")