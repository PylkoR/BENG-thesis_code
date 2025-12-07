import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

# Konfiguracja ścieżek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_lstm.csv')

# Katalogi wyjściowe
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'training_output')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'best_params.json')

# Artefakty (model i skalery)
MODEL_PATH = os.path.join(OUTPUT_DIR, 'lstm_model.keras')
SCALER_FEAT_PATH = os.path.join(OUTPUT_DIR, 'scaler_features.pkl')
SCALER_TARG_PATH = os.path.join(OUTPUT_DIR, 'scaler_target.pkl')
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, 'lstm_training_loss.png')
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, 'training_log.csv')

# Stałe
TEST_SPLIT = 0.1 

def create_sequences(X_data, y_data, seq_len):
    X, y = [], []
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i:(i + seq_len)])
        y.append(y_data[i + seq_len])
    return np.array(X), np.array(y)

try:
    # 1. Wczytanie konfiguracji z Tunera
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"Brak pliku {PARAMS_PATH}. Najpierw uruchom lstm_tune.py!")
        
    print(f"Wczytywanie parametrów z: {PARAMS_PATH}")
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    
    # Pobieramy kluczowe ustawienia
    SEQ_LEN = best_params['seq_len']
    BATCH_SIZE = best_params['batch_size']
    LEARNING_RATE = best_params['learning_rate']
    TARGET_COL = best_params.get('target_col', 'mwig40_Zamkniecie')
    SCALER_RANGE = tuple(best_params.get('scaler_range', [0, 1]))
    
    print(f"Cel predykcji: {TARGET_COL}")
    print(f"Zakres skalera: {SCALER_RANGE}")
    print(f"Architektura: SEQ_LEN={SEQ_LEN}, BATCH={BATCH_SIZE}, LR={LEARNING_RATE}")

    # 2. Wczytanie danych
    print(f"Wczytywanie danych z: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Usuwamy NaN (ważne przy zwrotach)
    df = df.dropna()
    feature_cols = df.columns.tolist()

    # 3. Podział danych
    train_size = int(len(df) * (1 - TEST_SPLIT))
    train_df = df.iloc[:train_size]
    # test_df = df.iloc[train_size:]

    print(f"Liczba wierszy treningowych: {len(train_df)}")

    # 4. Skalowanie
    print("Skalowanie danych...")
    scaler_features = MinMaxScaler(feature_range=SCALER_RANGE)
    scaler_target = MinMaxScaler(feature_range=SCALER_RANGE)

    # Fitting na danych treningowych
    scaled_train_features = scaler_features.fit_transform(train_df[feature_cols])
    scaled_train_target = scaler_target.fit_transform(train_df[[TARGET_COL]])

    # Skalery
    joblib.dump(scaler_features, SCALER_FEAT_PATH)
    joblib.dump(scaler_target, SCALER_TARG_PATH)
    print("Zapisano skalery.")

    # 5. Przygotowanie sekwencji
    X_train, y_train = create_sequences(scaled_train_features, scaled_train_target, SEQ_LEN)
    print(f"Wymiary danych wejściowych X: {X_train.shape}")

    # 6. Budowa modelu (dynamicznie z JSON)
    print("Budowa modelu...")
    model = Sequential()
    
    # Wejście
    model.add(Input(shape=(SEQ_LEN, len(feature_cols))))
    
    # Warstwy ukryte
    layers_config = best_params['layers']
    num_layers = len(layers_config)
    
    for i, layer_conf in enumerate(layers_config):
        units = layer_conf['units']
        dropout_rate = layer_conf['dropout']
        
        # Ostatni LSTM zawsze return_sequences=False
        is_last_lstm = (i == num_layers - 1)
        return_seq = not is_last_lstm
        
        model.add(LSTM(units=units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
        print(f" -> Warstwa LSTM: {units} neuronów, Dropout: {dropout_rate}")

    # Warstwy gęste (Dense)
    dense_units = best_params['dense_units']
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1)) # Wyjście
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

    # 7. Trening
    print("\nRozpoczynam trening...")
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    csv_logger = CSVLogger(LOG_CSV_PATH)

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=150,
        validation_split=0.2,
        callbacks=[early_stop, csv_logger],
        verbose=1
    )

    # 8. Zapis modelu i wykresu
    model.save(MODEL_PATH)
    print(f"\nModel zapisano w: {MODEL_PATH}")

    # Wykres Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    
    # Zaznaczamy najlepszy punkt
    best_epoch = np.argmin(history.history['val_loss'])
    best_loss = history.history['val_loss'][best_epoch]
    plt.scatter(best_epoch, best_loss, c='red', s=50, label=f'Best Epoch ({best_epoch})')

    plt.title(f'Krzywa uczenia (Min Val Loss: {best_loss:.6f})')
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Zapisano wykres: {LOSS_PLOT_PATH}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")