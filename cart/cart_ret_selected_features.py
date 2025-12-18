import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Pomiar czasu - start
start_time = time.time()

# Definicja ścieżek systemowych
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset_ret_lag.csv')

# Konfiguracja katalogów wyjściowych
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'cart_ret_output_selected_features')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definicja ścieżek plików wynikowych
OUTPUT_PLOT_FULL = os.path.join(OUTPUT_DIR, 'cart_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(OUTPUT_DIR, 'cart_forecast_zoom.png')
OUTPUT_PLOT_TREE = os.path.join(OUTPUT_DIR, 'cart_structure.png')
OUTPUT_PRED_PATH = os.path.join(OUTPUT_DIR, 'cart_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(OUTPUT_DIR, 'cart_metrics.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'cart_model.pkl')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'cart_best_params.json')

# Parametry globalne
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12

# Data podziału na treningowy i testowy
SPLIT_DATE = '2022-01-03'

try:
    # 1. Import danych
    print(f"Wczytywanie gotowych danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2. Selekcja cech
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
    
    print(f"\n--- SELEKCJA CECH ---")
    print(f"Zdefiniowano {len(SELECTED_FEATURES)} kluczowych cech.")
    
    # Sprawdzenie, czy wszystkie wybrane cechy istnieją w pliku
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_features = set(SELECTED_FEATURES) - set(available_features)
    
    if missing_features:
        print(f"UWAGA: Brakujące kolumny w pliku: {missing_features}")
    
    df = df.dropna()
    
    X = df[available_features]
    y = df[target_col_source]

    print(f"Ostateczna liczba cech w modelu: {len(X.columns)}")
    print(f"Cechy: {X.columns.tolist()}")

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

    # 4. Optymalizacja hiperparametrów i trening modelu
    print("\nInicjalizacja RandomizedSearchCV...")
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    param_dist = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'max_depth': [3, 4, 5, 8, 10, 15, 20, None],
        'min_samples_split': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'min_samples_leaf': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.001, 0.005, 0.01]
    }

    dt = DecisionTreeRegressor(random_state=42)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        dt, 
        param_distributions=param_dist, 
        n_iter=300,
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

    # 6. Transformacja zwrotów na ceny
    print("Rekonstrukcja poziomów cenowych...")
    if 'mwig40_Zamkniecie' not in test_df.columns:
        raise ValueError("Brak kolumny referencyjnej ceny zamknięcia.")

    actual_prices = test_df['mwig40_Zamkniecie'].values
    last_train_price = train_df['mwig40_Zamkniecie'].iloc[-1]
    
    # Rekonstrukcja: cena(t) = cena(t-1) * exp(zwrot)
    prev_prices = np.concatenate(([last_train_price], actual_prices[:-1]))
    predicted_prices = prev_prices * np.exp(pred_returns)
    
    forecast_series = pd.Series(predicted_prices, index=test_df.index)
    actual_series = test_df['mwig40_Zamkniecie']

    # 7. Obliczenie metryk błędu
    mae = mean_absolute_error(actual_series, forecast_series)
    rmse = np.sqrt(mean_squared_error(actual_series, forecast_series))
    mape = np.mean(np.abs((actual_series - forecast_series) / actual_series)) * 100
    
    actual_returns = test_df[target_col_source].values
    direction_match = np.sign(pred_returns) == np.sign(actual_returns)
    dir_acc = np.mean(direction_match) * 100

    # Pomiar czasu - koniec
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n--- WYNIKI MODELU CART (Selekcja Cech) ---")
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

    # Wykres 1: Pełny (koniec treningu + test)
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index[-200:], train_df['mwig40_Zamkniecie'].tail(200), label='Dane Treningowe')
    plt.plot(test_df.index, actual_series, label='Dane Rzeczywiste', color='green')
    plt.plot(test_df.index, forecast_series, label='Prognoza CART', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Prognoza CART vs Rzeczywistość (Wybrane Cechy)', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL, dpi=200)
    print(f"Zapisano wykres główny: {OUTPUT_PLOT_FULL}")

    # Wykres 2: Zoom
    start_date_zoom = '2025-01-02'
    
    # Filtrujemy dane do wykresu zoom
    plot_data_actual = test_df.loc[start_date_zoom:]
    plot_data_forecast = forecast_series.loc[start_date_zoom:]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_data_actual.index, plot_data_actual['mwig40_Zamkniecie'], label='Dane Rzeczywiste', linewidth=2, color='green')
    plt.plot(plot_data_forecast.index, plot_data_forecast, label='Prognoza CART', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Szczegóły prognozy od {start_date_zoom}', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Kurs mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM, dpi=200)
    print(f"Zapisano wykres zoom: {OUTPUT_PLOT_ZOOM}")
    
    plt.show()

    # Wizualizacja struktury drzewa (max_depth=4 dla czytelności)
    plt.figure(figsize=(20, 10))
    plot_tree(model, max_depth=4, feature_names=available_features, filled=True, fontsize=10, rounded=True)
    plt.title(f"Struktura Drzewa CART (Top Features)", fontsize=TITLE_FONT_SIZE)
    plt.savefig(OUTPUT_PLOT_TREE, dpi=200)

    # Eksport wyników
    results_df = pd.DataFrame({
        'Data': test_df.index,
        'Rzeczywiste': actual_series.values,
        'Prognoza_Cena': forecast_series.values,
        'Prognoza_Zwrot': pred_returns,
        'Rzeczywisty_Zwrot': test_df[target_col_source].values
    })
    results_df.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',', index=False)
    print(f"Zapisano wyniki predykcji: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")
    import traceback
    traceback.print_exc()