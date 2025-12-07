import pandas as pd
import numpy as np
import os

# Konfiguracja ścieżek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset.csv')

# Ścieżka wyjściowa
OUTPUT_DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_lstm.csv')

# Funkcje pomocnicze do wskaźników
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_percent_b(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = upper - lower
    width = width.replace(0, np.nan) 
    return (series - lower) / width

try:
    # 1. Wczytanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.set_index('Data').sort_index()

    # 2. Inżynieria Cech
    print("Obliczanie wskaźników...")
    
    # Logarytmiczne zwroty dla rynków zewnętrznych (synchronizacja dynamiki)
    external_assets = {
        'dax_Zamkniecie': 'DAX_Ret',
        'spx_Zamkniecie': 'SPX_Ret',
        'nkx_Zamkniecie': 'NKX_Ret',
        'brent_Zamkniecie': 'Brent_Ret',
        'eurpln_Zamkniecie': 'EURPLN_Ret',
        'usdpln_Zamkniecie': 'USDPLN_Ret',
        'wig20_Zamkniecie': 'WIG20_Ret'
    }

    for col, new_name in external_assets.items():
        if col in df.columns:
            df[new_name] = np.log(df[col] / df[col].shift(1))

    # Wskaźniki dla celu (mWIG40)
    target_col = 'mwig40_Zamkniecie'
    
    # Log Return samego mWIG40
    df['mWIG40_Ret'] = np.log(df[target_col] / df[target_col].shift(1))

    # RSI (Momentum)
    df['RSI_14'] = calculate_rsi(df[target_col], window=14)

    # Bollinger %B (Zmienność relatywna)
    df['Bollinger_PB'] = calculate_bollinger_percent_b(df[target_col])

    # MACD Histogram (Trend)
    exp12 = df[target_col].ewm(span=12, adjust=False).mean()
    exp26 = df[target_col].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal

    # Zmienność historyczna (20 dni)
    df['Volatility_20'] = df['mWIG40_Ret'].rolling(window=20).std()

    # 3. Selekcja i czyszczenie
    features_to_keep = [
        'mwig40_Zamkniecie',    # Cel predykcji
        'mwig40_Wolumen',       
        'mWIG40_Ret',           
        'RSI_14', 'Bollinger_PB', 'MACD_Hist', 'Volatility_20', 
        'DAX_Ret', 'SPX_Ret', 'NKX_Ret', 'Brent_Ret', 'EURPLN_Ret', 'USDPLN_Ret', 'WIG20_Ret',
        'vix_Zamkniecie'        
    ]

    # Zabezpieczenie przed brakującymi kolumnami
    available_features = [f for f in features_to_keep if f in df.columns]
    df_processed = df[available_features].copy()

    # Usuwam NaN powstałe przy liczeniu wskaźników (pierwsze ~20-50 wierszy)
    initial_len = len(df_processed)
    df_processed = df_processed.dropna()
    print(f"Usunięto {initial_len - len(df_processed)} wierszy (rozruch wskaźników)")

    # 4. Zapis
    df_processed.to_csv(OUTPUT_DATA_PATH, sep=';', decimal=',')
    print(f"Zapisano przetworzone dane do: {OUTPUT_DATA_PATH}")

except Exception as e:
    print(f"Wystąpił błąd krytyczny: {e}")