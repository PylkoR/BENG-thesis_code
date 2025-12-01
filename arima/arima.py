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
OUTPUT_PLOT_FULL = os.path.join(SCRIPT_DIR, 'arima_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(SCRIPT_DIR, 'arima_forecast_zoom.png')
OUTPUT_PLOT_DIAG = os.path.join(SCRIPT_DIR, 'arima_diagnostics.png')
OUTPUT_PRED_PATH = os.path.join(SCRIPT_DIR, 'arima_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(SCRIPT_DIR, 'arima_metrics.csv')

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
    data = df[[target_col]].copy()

    # 2. Podział na zbiór treningowy i testowy (90/10)
    train_size = int(len(data) * 0.9)
    train, test = data.iloc[:train_size], data.iloc[train_size:].copy()
    
    print(f"Zbiór treningowy: {len(train)} obserwacji")
    print(f"Zbiór testowy: {len(test)} obserwacji")

    # 3. Dobór parametrów i trening modelu (Auto ARIMA)
    print("\nRozpoczynam automatyczny dobór parametrów ARIMA...")
    
    model = pm.auto_arima(train[target_col], 
                          start_p=0, start_q=0,
                          max_p=5, max_q=5,
                          d=None,           
                          seasonal=False,   
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)

    print(f"\nWybrano optymalny model: {model.order}")

    # Wygenerowanie wykresów diagnostycznych modelu (reszty, normalność)
    print("Generowanie diagnostyki modelu...")
    model.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIAG)
    plt.close() # Zamknięcie, aby nie nakładać na kolejne wykresy

    # 4. Prognoza krocząca (Rolling Forecast)
    print("\nRozpoczynam prognozę kroczącą...")
    
    forecasts = []
    rolling_model = model # Kopia referencji do modelu

    for i in range(len(test)):
        # Predykcja na jeden krok w przód
        prediction = rolling_model.predict(n_periods=1)
        
        # Ekstrakcja wartości liczbowej z wyniku predykcji
        if isinstance(prediction, pd.Series):
            fc = float(prediction.iloc[0])
        else:
            fc = float(prediction[0])
            
        forecasts.append(fc)
        
        # Aktualizacja modelu o rzeczywistą wartość (obserwacja staje się znana)
        true_value = test[target_col].iloc[i]
        rolling_model.update([true_value])
        
        # Monitorowanie postępu
        if i % 50 == 0:
            print(f"Postęp: {i}/{len(test)} kroków")

    forecast_series = pd.Series(forecasts, index=test.index)

    # 5. Obliczenie metryk błędu
    mae = mean_absolute_error(test[target_col], forecast_series)
    rmse = np.sqrt(mean_squared_error(test[target_col], forecast_series))
    mape = np.mean(np.abs((test[target_col] - forecast_series) / test[target_col])) * 100

    print(f"\nWyniki modelu ARIMA:")
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

    # 6. Wizualizacja wyników

    # Wykres 1: Szerszy kontekst (ostatnie 200 dni treningu + test)
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-200:], train[target_col].iloc[-200:], label='Dane Treningowe')
    plt.plot(test.index, test[target_col], label='Dane Rzeczywiste')
    plt.plot(test.index, forecast_series, label='Prognoza ARIMA', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'Prognoza ARIMA vs Rzeczywistość (Model {model.order})', fontsize=TITLE_FONT_SIZE)
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
    plt.plot(last_100_days, forecast_series.tail(100), label='Prognoza ARIMA', color='red', linestyle='--', linewidth=2)

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

    # 7. Zapis wyników predykcji
    results_df = test.copy()
    results_df['ARIMA_Forecast'] = forecast_series
    results_df.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',')
    print(f"Zapisano prognozy do: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Wystąpił błąd krytyczny: {e}")