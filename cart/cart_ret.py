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

# Konfiguracja katalogów wyjściowych
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'cart_ret_output')
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
TEST_SPLIT = 0.1
N_LAGS = 5 

try:
    # 1. Import i wstępne przetwarzanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2. Inżynieria cech (Automatyzacja dla wszystkich kolumn)
    target_col_source = 'mWIG40_Ret'
    
    print(f"Generowanie cech opóźnionych (Lags={N_LAGS}) dla wszystkich kolumn numerycznych...")
    
    # Identyfikacja kolumn numerycznych
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Iteracja po wszystkich kolumnach i tworzenie opóźnień
    for col in numeric_cols:
        for i in range(1, N_LAGS + 1):
            df[f'{col}_Lag_{i}'] = df[col].shift(i)
    
    # Usunięcie wierszy z brakującymi danymi (NaN) wynikłymi z przesunięcia
    df = df.dropna()

    # Zapis przetworzonego zbioru danych (dane wejściowe + wszystkie lagi)
    print(f"Zapisywanie rozszerzonego datasetu do: {PROCESSED_DATA_PATH}")
    df.to_csv(PROCESSED_DATA_PATH, sep=';', decimal=',')

    # Definicja zbioru cech (X) - wybór tylko kolumn z sufiksem '_Lag_'
    feature_cols = [c for c in df.columns if '_Lag_' in c]
    X = df[feature_cols]
    y = df[target_col_source]

    print(f"Liczba wygenerowanych cech wejściowych: {len(feature_cols)}")

    # 3. Podział zbioru na treningowy i testowy
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Wyodrębnienie pełnych ramek danych dla celów wizualizacji i rekonstrukcji cen
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Zbiór treningowy: {len(train_df)} dni")
    print(f"Zbiór testowy: {len(test_df)} dni")

    # 4. Optymalizacja hiperparametrów i trening modelu
    print("\nInicjalizacja GridSearch...")
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse'],
        'max_depth': [3, 4, 5, 6, 10, 15, None],
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
    print(f"\nWybrane parametry optymalne: {grid_search.best_params_}")
    
    # Eksport modelu i parametrów
    joblib.dump(model, MODEL_PATH)
    pd.Series(grid_search.best_params_).to_json(PARAMS_PATH)

    # 5. Generowanie predykcji
    print("Obliczanie prognoz dla zbioru testowego...")
    pred_returns = model.predict(X_test)

    # 6. Transformacja zwrotów na ceny
    print("Rekonstrukcja poziomów cenowych...")
    
    if 'mwig40_Zamkniecie' not in test_df.columns:
        raise ValueError("Brak kolumny referencyjnej ceny zamknięcia.")

    actual_prices = test_df['mwig40_Zamkniecie'].values
    last_train_price = train_df['mwig40_Zamkniecie'].iloc[-1]
    
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

    print(f"\n--- WYNIKI MODELU CART ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f} %")
    print(f"Dir Accuracy: {dir_acc:.2f} %")

    # Zapis metryk
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy'],
        'Value': [mae, rmse, mape, dir_acc]
    })
    metrics_df.to_csv(OUTPUT_METRICS_PATH, sep=';', decimal=',', index=False)

    # 8. Wizualizacja
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index[-200:], train_df['mwig40_Zamkniecie'].tail(200), label='Historia (Trening)')
    plt.plot(test_df.index, actual_series, label='Rzeczywistość')
    plt.plot(test_df.index, forecast_series, label='Prognoza CART', color='green', linestyle='--', alpha=0.8)
    
    plt.title('CART: Prognoza Ceny', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_FULL)
    print(f"Zapisano wykres główny: {OUTPUT_PLOT_FULL}")

    plt.figure(figsize=(12, 6))
    last_100 = test_df.index[-100:]
    plt.plot(last_100, actual_series.tail(100), label='Rzeczywistość', linewidth=2)
    plt.plot(last_100, forecast_series.tail(100), label='Prognoza CART', color='green', linestyle='--', linewidth=2)
    
    plt.title('Szczegóły prognozy (ostatnie 100 dni)', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize=AXIS_FONT_SIZE)
    plt.ylabel('mWIG40', fontsize=AXIS_FONT_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT_ZOOM)
    print(f"Zapisano wykres zoom: {OUTPUT_PLOT_ZOOM}")

    # Wizualizacja struktury drzewa (z ograniczeniem głębokości dla czytelności)
    plt.figure(figsize=(20, 10))
    plot_tree(model, max_depth=4, feature_names=feature_cols, filled=True, fontsize=10, rounded=True)
    plt.title(f"Struktura Drzewa (Top 4 poziomy) | Max Depth: {model.max_depth}", fontsize=TITLE_FONT_SIZE)
    plt.savefig(OUTPUT_PLOT_TREE)
    print(f"Zapisano strukturę drzewa: {OUTPUT_PLOT_TREE}")

    # Eksport wyników predykcji
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