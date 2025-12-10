import pandas as pd
import numpy as np
import os
import json
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_lstm.csv')

# Katalogi wyjściowe
TUNER_DIR = os.path.join(SCRIPT_DIR, 'tuning_cache')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'training_output')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'best_params.json')

os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parametry - zmiana
TARGET_COL = 'mWIG40_Ret'
SEQ_LEN = 60         
SPLIT_DATE = '2022-01-03'
VAL_SPLIT = 0.2     
BATCH_SIZE = 32     

# Funkcje
def create_sequences(X_data, y_data, seq_len):
    X, y = [], []
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i:(i + seq_len)])
        y.append(y_data[i + seq_len])
    return np.array(X), np.array(y)

def build_model(hp):
    model = Sequential()
    
    # Warstwa wejściowa
    model.add(Input(shape=(SEQ_LEN, n_features)))
    
    # 1. Strojenie: Liczba warstw LSTM
    num_layers = hp.Int('num_layers', 1, 2)
    
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=128, step=32)
        return_seq = True if i < num_layers - 1 else False
        
        model.add(LSTM(units=units, return_sequences=return_seq))
        
        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout))
    
    # Warstwa gęsta
    model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    
    # Wyjście
    model.add(Dense(1)) # Przewidujemy jedną wartość (zwrot w dniu t+1)
    
    # Strojenie Learning Rate
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

# Tuner
try:
    # 1. Wczytanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()

    feature_cols = df.columns.tolist()
    n_features = len(feature_cols)
    print(f"Cel predykcji: {TARGET_COL}")
    print(f"Liczba cech wejściowych: {n_features}")

    # 2. Podział danych
    train_test_split_idx = int(len(df) * (1 - TEST_SPLIT))
    train_val_df = df.iloc[:train_test_split_idx]
    
    # 3. Skalowanie
    print("Skalowanie danych (zakres -1 do 1)...")
    scaler_X = MinMaxScaler((-1, 1)).fit(train_val_df[feature_cols])
    scaler_y = MinMaxScaler((-1, 1)).fit(train_val_df[[TARGET_COL]])

    X_scaled = scaler_X.transform(train_val_df[feature_cols])
    y_scaled = scaler_y.transform(train_val_df[[TARGET_COL]])

    # Tworzenie sekwencji
    print("Tworzenie sekwencji czasowych...")
    X, y = create_sequences(X_scaled, y_scaled, SEQ_LEN)

    # 4. Uruchomienie Tunera
    print("\nInicjalizacja Keras Tuner...")
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,           
        executions_per_trial=1,  
        directory=TUNER_DIR,
        project_name='mwig40_lstm_ret',
        overwrite=True           
    )

    tuner.search_space_summary()

    print("\nRozpoczynam poszukiwanie najlepszych hiperparametrów...")
    stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    tuner.search(
        X, y,
        epochs=15,              
        validation_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        callbacks=[stop_early],
        verbose=1
    )

    # 5. Zapis wyników
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n--- ZNALEZIONO NAJLEPSZĄ KONFIGURACJĘ (dla zwrotów) ---")
    
    best_params = {
        'target_col': TARGET_COL, # Zapisujemy, co było celem, żeby inne skrypty wiedziały
        'scaler_range': [-1, 1],  # Zapisujemy zakres skalowania
        'num_layers': best_hps.get('num_layers'),
        'dense_units': best_hps.get('dense_units'),
        'learning_rate': best_hps.get('learning_rate'),
        'batch_size': BATCH_SIZE,
        'seq_len': SEQ_LEN,
        'layers': []
    }

    for i in range(best_params['num_layers']):
        layer_config = {
            'units': best_hps.get(f'units_{i}'),
            'dropout': best_hps.get(f'dropout_{i}')
        }
        best_params['layers'].append(layer_config)
        print(f"Warstwa {i+1}: {layer_config}")

    with open(PARAMS_PATH, 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print(f"\nZapisano parametry do: {PARAMS_PATH}")
    print("Krok 1 zakończony. Możesz przejść do treningu.")

except Exception as e:
    print(f"Błąd krytyczny: {e}")