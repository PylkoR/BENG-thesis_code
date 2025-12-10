import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Konfiguracja ścieżek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset.csv')

# Ścieżki wyjściowe
OUTPUT_PLOT_DIAG = os.path.join(SCRIPT_DIR, 'arima_diagnostics.png')

# Data poodziału na treningowy i testowy
SPLIT_DATE = '2022-01-03'

try:
    # 1. Wczytanie i przygotowanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.set_index('Data').sort_index()

    target_col = 'mwig40_Zamkniecie'
    data = df[[target_col]].copy()

    # 2. Podział na zbiór treningowy i testowy (90/10)
    mask_train = data.index < SPLIT_DATE
    mask_test = data.index >= SPLIT_DATE

    train = data.loc[mask_train]
    test = data.loc[mask_test]
    
    print(f"Zbiór treningowy: {len(train)} obserwacji")
    print(f"Zbiór testowy: {len(test)} obserwacji")

    # 3. Dobór parametrów i trening modelu (Auto ARIMA)
    print("\nAutomatyczny dobór parametrów ARIMA...")
    
    model = pm.auto_arima(train[target_col], 
                          start_p=0, start_q=0,
                          max_p=5, max_q=5,
                          d=None,           
                          seasonal=False,   
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=False)

    print(f"\nWybrano optymalny model: {model.order}")

    # Wygenerowanie wykresów diagnostycznych modelu (reszty, normalność)
    print("Generowanie diagnostyki modelu...")
    model.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIAG)
    plt.close()

except Exception as e:
    print(f"Wystąpił błąd krytyczny: {e}")