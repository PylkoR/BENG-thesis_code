import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Pomiar czasu - start
start_time = time.time()

# Definicja ścieżek systemowych
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset_ret_lag.csv')

# Konfiguracja katalogów wyjściowych
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'xgboost_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definicja ścieżek plików wynikowych
OUTPUT_PLOT_FULL = os.path.join(OUTPUT_DIR, 'xgboost_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(OUTPUT_DIR, 'xgboost_forecast_zoom.png')
OUTPUT_PLOT_IMP = os.path.join(OUTPUT_DIR, 'xgboost_importance.png')
OUTPUT_PRED_PATH = os.path.join(OUTPUT_DIR, 'xgboost_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(OUTPUT_DIR, 'xgboost_metrics.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'xgboost_model.pkl')

# Parametry globalne
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12

# Data podziału na treningowy i testowy
SPLIT_DATE = '2022-01-03'

try:
    # 1. Import danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2. Selekcja cech (identyczna jak w CART dla porównywalności)
    target_col_source = 'mWIG40_Ret'
    
    SELECTED_FEATURES = [
        'SPX_Ret_Lag_1',
        'mWIG40_Ret_Lag_1',
        'EURPLN_Ret_Lag_1',
        'WIG20_Ret_Lag_1',
        'Bollinger_PB_Lag_1',
        'DAX_Ret_Lag_1',
        'RSI_14_Lag_1',
        'USDPLN_Ret_Lag_1',
        'MACD_Hist_Lag_1',
        'NKX_Ret_Lag_2',
        'Brent_Ret_Lag_3',
        'vix_Zamkniecie_Lag_3',
        'mWIG40_Ret_Lag_2',
        'mWIG40_Ret_Lag_6',
        'SPX_Ret_Lag_3',
        'SPX_Ret_Lag_6',
        'Bollinger_PB_Lag_6',
        'Volatility_20_Lag_13',
        'Volatility_20_Lag_2',
        'RSI_14_Lag_2'
    ]
    
    print(f"\n--- PRZYGOTOWANIE DANYCH ---")
    # Sprawdzenie dostępności cech
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    df = df.dropna()
    
    X = df[available_features]
    y = df[target_col_source]

    print(f"Liczba cech: {len(X.columns)}")

    # 3. Podział zbioru na treningowy i testowy
    print(f"\nDzielenie zbioru danych względem daty: {SPLIT_DATE}")
    
    mask_train = X.index < SPLIT_DATE
    mask_test = X.index >= SPLIT_DATE

    X_train, X_test = X.loc[mask_train], X.loc[mask_test]
    y_train, y_test = y.loc[mask_train], y.loc[mask_test]
    
    train_df = df.loc[mask_train]
    test_df = df.loc[mask_test]

    print(f"Zbiór treningowy: {len(train_df)} obserwacji")
    print(f"Zbiór testowy: {len(test_df)} obserwacji")

    # 4. Optymalizacja hiperparametrów (RandomizedSearchCV)
    print("\nInicjalizacja tuningu XGBoost...")
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    # Siatka parametrów dla XGBoost
    param_dist = {
        'n_estimators': [500, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'learning_rate': [0.001, 0.005, 0.01, 0.015, 0.02],
        'max_depth': [4, 5, 6, 7, 8],
        'min_child_weight': [5, 10, 20, 30],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.5, 1, 1.5, 2]
    }

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )

    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=200,  # Liczba iteracji
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_
    print(f"\nWybrane parametry optymalne: {random_search.best_params_}")
    
    joblib.dump(model, MODEL_PATH)

    # 5. Generowanie predykcji
    pred_returns = model.predict(X_test)

    # 6. Rekonstrukcja cen (logika identyczna jak w CART)
    print("Rekonstrukcja poziomów cenowych...")
    actual_prices = test_df['mwig40_Zamkniecie'].values
    last_train_price = train_df['mwig40_Zamkniecie'].iloc[-1]
    
    # One-step-ahead reconstruction
    prev_prices = np.concatenate(([last_train_price], actual_prices[:-1]))
    predicted_prices = prev_prices * np.exp(pred_returns)
    
    forecast_series = pd.Series(predicted_prices, index=test_df.index)
    actual_series = test_df['mwig40_Zamkniecie']

    # 7. Obliczenie metryk
    mae = mean_absolute_error(actual_series, forecast_series)
    rmse = np.sqrt(mean_squared_error(actual_series, forecast_series))
    mape = np.mean(np.abs((actual_series - forecast_series) / actual_series)) * 100
    
    # Directional Accuracy
    actual_returns = test_df[target_col_source].values
    direction_match = np.sign(pred_returns) == np.sign(actual_returns)
    dir_acc = np.mean(direction_match) * 100

    # Pomiar czasu - koniec
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n--- WYNIKI MODELU XGBoost ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.4f} %")
    print(f"Dir Accuracy: {dir_acc:.2f} %")
    print(f"Czas wykonania: {execution_time:.4f} s")

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy_Pct', 'Execution_Time_Sec'],
        'Value': [mae, rmse, mape, dir_acc, execution_time]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)

    # 8. Wizualizacja

    # Wykres 1: Pełny
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index[-200:], train_df['mwig40_Zamkniecie'].tail(200), label='Dane Treningowe')
    plt.plot(test_df.index, actual_series, label='Dane Rzeczywiste', color='green')
    plt.plot(test_df.index, forecast_series, label='Prognoza XGBoost', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Prognoza XGBoost vs Rzeczywistość', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL, dpi=200)
    print(f"Zapisano wykres główny: {OUTPUT_PLOT_FULL}")

    # Wykres 2: Zoom
    start_date_zoom = '2025-01-02'
    plot_data_actual = test_df.loc[start_date_zoom:]
    plot_data_forecast = forecast_series.loc[start_date_zoom:]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_data_actual.index, plot_data_actual['mwig40_Zamkniecie'], label='Dane Rzeczywiste', linewidth=2, color='green')
    plt.plot(plot_data_forecast.index, plot_data_forecast, label='Prognoza XGBoost', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Szczegóły prognozy od {start_date_zoom}', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM, dpi=200)
    print(f"Zapisano wykres zoom: {OUTPUT_PLOT_ZOOM}")

    # Wykres 3: Ważność cech (Feature Importance)
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, height=0.5, title='Ważność cech (XGBoost)')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_IMP, dpi=200)
    print(f"Zapisano wykres ważności cech: {OUTPUT_PLOT_IMP}")

    plt.show()

    # Eksport wyników
    results_df = pd.DataFrame({
        'Data': test_df.index,
        'Rzeczywiste': actual_series.values,
        'Prognoza_Cena': forecast_series.values,
        'Prognoza_Zwrot': pred_returns,
        'Rzeczywisty_Zwrot': actual_returns
    })
    results_df.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',', index=False)
    print(f"Zapisano wyniki predykcji: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")
    import traceback
    traceback.print_exc()