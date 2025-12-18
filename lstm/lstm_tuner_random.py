import pandas as pd
import numpy as np
import os
import json
import joblib
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_ret.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'lstm_output')
TUNER_DIR = os.path.join(OUTPUT_DIR, 'tuning_cache_random')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'lstm_best_params.json')
FEATURES_PATH = os.path.join(OUTPUT_DIR, 'lstm_selected_features.json')
SCALER_X_PATH = os.path.join(OUTPUT_DIR, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(OUTPUT_DIR, 'scaler_y.pkl')

os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ustawienia
TARGET_COL = 'mWIG40_Ret'
SPLIT_DATE = '2022-01-03'
MAX_EPOCHS = 15
VAL_SPLIT = 0.2

# Ustawienia Random Search
MAX_TRIALS = 50 # Liczba kombinacji
EXECUTIONS_PER_TRIAL = 1  # Ile razy trenować każdą kombinację

SEQ_LEN = 60     
BATCH_SIZE = 32    

# Wybrane cechy
MY_FEATURES = [
    'mWIG40_Ret', 
    'RSI_14', 
    'Bollinger_PB',
    #'MACD_Hist',
    'Volatility_20',
    'SPX_Ret',
    #'DAX_Ret',
    #'WIG20_Ret',
    'NKX_Ret',
    #'Brent_Ret',
    'EURPLN_Ret',
    #'vix_Zamkniecie'
]

n_features_global = len(MY_FEATURES)

def create_sequences(X_data, y_data, seq_len):
    X, y = [], []
    if len(X_data) <= seq_len: return np.array(X), np.array(y)
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i:(i+seq_len)])
        y.append(y_data[i+seq_len])
    return np.array(X), np.array(y)

def build_model(hp):
    model = Sequential()
    # SEQ_LEN stałe
    model.add(Input(shape=(SEQ_LEN, n_features_global)))
    
    num_layers = hp.Int('num_lstm_layers', 1, 3)
    for i in range(num_layers):
        units = hp.Int(f'lstm_units_{i}', 32, 256, step=32)
        drop = hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)
        
        ret_seq = (i < num_layers - 1)
        
        model.add(LSTM(units=units, return_sequences=ret_seq))
        if drop > 0: model.add(Dropout(drop))
    
    if hp.Boolean('use_dense'):
        dense_units = hp.Int('dense_units', 16, 128, step=16)
        act = hp.Choice('dense_activation', ['relu', 'tanh', 'swish'])
        model.add(Dense(dense_units, activation=act))
        
    model.add(Dense(1))
    
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    opt_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    
    if opt_choice == 'adam': opt = Adam(learning_rate=lr)
    elif opt_choice == 'rmsprop': opt = RMSprop(learning_rate=lr)
    else: opt = SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    print(f"--- START TUNERA (RANDOM SEARCH) ---")
    print(f"Konfiguracja: SEQ={SEQ_LEN}, BATCH={BATCH_SIZE}")
    
    # 1. Dane
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    
    missing = [f for f in MY_FEATURES if f not in df.columns]
    if missing: raise ValueError(f"Brak kolumn: {missing}")
    
    # 2. Skalowanie
    mask_train = df.index < SPLIT_DATE
    df_train = df.loc[mask_train].copy()
    
    X_train = df_train[MY_FEATURES]
    y_train = df_train[[TARGET_COL]]
    
    scaler_X = MinMaxScaler((-1, 1))
    scaler_y = MinMaxScaler((-1, 1))
    
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train)
    
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    with open(FEATURES_PATH, 'w') as f: json.dump(MY_FEATURES, f, indent=4)
    
    # 3. Sekwencje
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)
    
    # 4. Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=TUNER_DIR,
        project_name='lstm_random_config',
        overwrite=True
    )
    
    stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    tuner.search(
        X_seq, y_seq, 
        epochs=MAX_EPOCHS,
        validation_split=VAL_SPLIT, 
        callbacks=[stop_early], 
        verbose=1,
        batch_size=BATCH_SIZE
    )
    
    # 5. Zapis wyników (Eksport konfiguracji)
    best_hps = tuner.get_best_hyperparameters()[0]
    num_layers = best_hps.get('num_lstm_layers')
    
    # JSON
    layers_config = []
    for i in range(num_layers):
        layers_config.append({
            'units': best_hps.get(f'lstm_units_{i}'),
            'dropout': best_hps.get(f'dropout_{i}'),
            'return_sequences': (i < num_layers - 1)
        })

    params_export = {
        'meta': {
            'target': TARGET_COL, 
            'split': SPLIT_DATE
        },
        'structure': {
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'learning_rate': best_hps.get('lr'),
            'optimizer': best_hps.get('optimizer'),
            'use_dense': best_hps.get('use_dense'),
            'dense_units': best_hps.get('dense_units') if best_hps.get('use_dense') else 0,
            'dense_activation': best_hps.get('dense_activation') if best_hps.get('use_dense') else 'relu'
        },
        'layers_config': layers_config
    }
    
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params_export, f, indent=4)

    print(f"\nPełna konfiguracja zapisana w: {PARAMS_PATH}")