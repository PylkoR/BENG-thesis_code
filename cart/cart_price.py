import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Definicja ścieżek systemowych
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset_lstm.csv')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset_cart.csv')

# Konfiguracja katalogów wyjściowych (zmieniona nazwa folderu)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'cart_price_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definicja ścieżek plików wynikowych
OUTPUT_PLOT_FULL = os.path.join(OUTPUT_DIR, 'cart_price_forecast_full.png')
OUTPUT_PLOT_ZOOM = os.path.join(OUTPUT_DIR, 'cart_price_forecast_zoom.png')
OUTPUT_PLOT_TREE = os.path.join(OUTPUT_DIR, 'cart_price_structure.png')
OUTPUT_PRED_PATH = os.path.join(OUTPUT_DIR, 'cart_price_predictions.csv')
OUTPUT_METRICS_PATH = os.path.join(OUTPUT_DIR, 'cart_price_metrics.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'cart_price_model.pkl')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'cart_price_best_params.json')

# Parametry globalne
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TEST_SPLIT = 0.1
N_LAGS = 5 

try:
    # 1. Import i wstępne przetwarzanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2. Inżynieria cech
    target_col_source = 'mwig40_Zamkniecie'
    
    if target_col_source not in df.columns:
        raise ValueError(f"Brak kolumny celu: {target_col_source}")

    print(f"Generowanie cech opóźnionych (Lags={N_LAGS}) dla wszystkich kolumn numerycznych...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        for i in range(1, N_LAGS + 1):
            df[f'{col}_Lag_{i}'] = df[col].shift(i)
    
    df = df.dropna()

    print(f"Zapisywanie danych (Price Target) do: {PROCESSED_DATA_PATH}")
    df.to_csv(PROCESSED_DATA_PATH, sep=';', decimal=',')

    # Definicja X i y
    feature_cols = [c for c in df.columns if '_Lag_' in c]
    X = df[feature_cols]
    y = df[target_col_source] 

    print(f"Liczba cech: {len(feature_cols)}")

    # 3. Podział zbioru
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Trening: {len(train_df)}, Test: {len(test_df)}")

    # 4. Grid Search
    print("\nInicjalizacja GridSearch (Model Ceny)...")
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse'],
        'max_depth': [3, 4, 5, 6, 8, 10, 15, None],
        'min_samples_split': [10, 20, 30, 40, 50, 80],
        'min_samples_leaf': [10, 20, 30, 40, 50, 80],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    dt = DecisionTreeRegressor(random_state=42)
    
    grid_search = GridSearchCV(
        dt, 
        param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    print(f"\nNajlepsze parametry: {grid_search.best_params_}")
    
    joblib.dump(model, MODEL_PATH)
    pd.Series(grid_search.best_params_).to_json(PARAMS_PATH)

    # 5. Predykcja
    print("Prognozowanie cen...")
    pred_prices = model.predict(X_test)
    
    forecast_series = pd.Series(pred_prices, index=test_df.index)
    actual_series = test_df[target_col_source]

    # 6. Metryki
    mae = mean_absolute_error(actual_series, forecast_series)
    rmse = np.sqrt(mean_squared_error(actual_series, forecast_series))
    mape = np.mean(np.abs((actual_series - forecast_series) / actual_series)) * 100
    
    # Obliczanie trafności kierunku dla modelu cenowego
    # Kierunek = (Predykcja_t - Cena_t-1) vs (Rzeczywista_t - Cena_t-1)
    lag_col_name = f'{target_col_source}_Lag_1'
    if lag_col_name in X_test.columns:
        prev_prices_test = X_test[lag_col_name]
        
        # Rzeczywista zmiana
        actual_delta = y_test - prev_prices_test
        # Prognozowana zmiana
        pred_delta = pred_prices - prev_prices_test
        
        direction_match = np.sign(actual_delta) == np.sign(pred_delta)
        dir_acc = np.mean(direction_match) * 100
    else:
        print("UWAGA: Brak kolumny Lag_1 do obliczenia Direction Accuracy.")
        dir_acc = 0.0

    print(f"\n--- WYNIKI MODELU CART (DIRECT PRICE) ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f} %")
    print(f"Dir Accuracy: {dir_acc:.2f} %")

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy'],
        'Value': [mae, rmse, mape, dir_acc]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)

    # 7. Wizualizacja
    
    # Wykres Full
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index[-200:], train_df[target_col_source].tail(200), label='Historia')
    plt.plot(test_df.index, actual_series, label='Rzeczywistość')
    plt.plot(test_df.index, forecast_series, label='Prognoza CART (Price)', color='red', linestyle='--', alpha=0.8)
    plt.title('CART: Bezpośrednia Prognoza Ceny', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('Cena', fontsize=AXIS_FONT_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL)

    # Wykres Zoom
    plt.figure(figsize=(12, 6))
    last_100 = test_df.index[-100:]
    plt.plot(last_100, actual_series.tail(100), label='Rzeczywistość', linewidth=2)
    plt.plot(last_100, forecast_series.tail(100), label='Prognoza CART (Price)', color='red', linestyle='--', linewidth=2)
    plt.title('Zoom (100 dni) - Model Cenowy', fontsize=TITLE_FONT_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM)

    # Wykres Drzewa
    plt.figure(figsize=(24, 12))
    plot_tree(
        model, 
        max_depth=4, 
        feature_names=feature_cols, 
        filled=True, 
        fontsize=9, 
        rounded=True, 
        precision=1
    )
    plt.title(f"Struktura Drzewa Cenowego", fontsize=TITLE_FONT_SIZE)
    plt.savefig(OUTPUT_PLOT_TREE)
    print(f"Zapisano strukturę drzewa: {OUTPUT_PLOT_TREE}")

    # Zapis wyników
    results_df = pd.DataFrame({
        'Data': test_df.index,
        'Rzeczywiste': actual_series.values,
        'Prognoza_Cena': forecast_series.values,
        'Diff': actual_series.values - forecast_series.values
    })
    results_df.to_csv(OUTPUT_PRED_PATH, sep=';', decimal=',', index=False)
    print(f"Zapisano wyniki: {OUTPUT_PRED_PATH}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")
    import traceback
    traceback.print_exc()