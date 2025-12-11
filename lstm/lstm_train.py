import pandas as pd
import numpy as np
import os
import json
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- KONFIGURACJA ŚCIEŻEK ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_ROOT = os.path.join(SCRIPT_DIR, 'lstm_output')
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_lstm.csv')

# Wejście (z Tunera)
PARAMS_PATH = os.path.join(LSTM_ROOT, 'lstm_best_params.json')
SCALER_X_PATH = os.path.join(LSTM_ROOT, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(LSTM_ROOT, 'scaler_y.pkl')

# Wyjście (Trening)
TRAIN_DIR = os.path.join(LSTM_ROOT, 'training_results')
os.makedirs(TRAIN_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(TRAIN_DIR, 'best_lstm_model.keras')

# --- FUNKCJE ---
def create_sequences(X_data, y_data, seq_len):
    X, y = [], []
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i:(i + seq_len)])
        y.append(y_data[i + seq_len])
    return np.array(X), np.array(y)

def build_model_from_config(config, n_features):
    struct = config['optimal_structure']
    layers_cfg = config['layers_config']
    
    model = Sequential()
    model.add(Input(shape=(struct['seq_len'], n_features)))
    
    # Warstwy LSTM
    num_layers = struct['num_lstm_layers']
    for i in range(num_layers):
        units = layers_cfg[i]['units']
        dropout = layers_cfg[i]['dropout']
        return_seq = True if i < num_layers - 1 else False
        
        model.add(LSTM(units=units, return_sequences=return_seq))
        if dropout > 0:
            model.add(Dropout(dropout))
            
    # Dense
    if struct.get('dense_units', 0) > 0:
        model.add(Dense(struct['dense_units'], activation='relu'))
        
    model.add(Dense(1))
    
    # Kompilacja
    lr = struct['learning_rate']
    opt_name = struct['optimizer']
    
    if opt_name == 'adam': optimizer = Adam(learning_rate=lr)
    elif opt_name == 'rmsprop': optimizer = RMSprop(learning_rate=lr)
    else: optimizer = SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Wczytanie konfiguracji
        print(f"Loading config: {PARAMS_PATH}")
        with open(PARAMS_PATH, 'r') as f:
            config = json.load(f)
        
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        
        SEQ_LEN = config['optimal_structure']['seq_len']
        BATCH_SIZE = config['optimal_structure']['batch_size']
        SPLIT_DATE = config['meta']['split_date']
        TARGET_COL = config['meta']['target_col']

        # 2. Dane
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()
        
        feature_cols = [c for c in df.columns if c != TARGET_COL]
        
        # Filtracja tylko zbioru treningowego
        mask_train = df.index < SPLIT_DATE
        df_train = df.loc[mask_train]
        
        # Transformacja
        X_train_scaled = scaler_X.transform(df_train[feature_cols])
        y_train_scaled = scaler_y.transform(df_train[[TARGET_COL]])
        
        X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
        print(f"Training set shape: {X_train.shape}")

        # 3. Trening
        model = build_model_from_config(config, n_features=len(feature_cols))
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='loss')
        ]
        
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Model saved to: {MODEL_SAVE_PATH}")

    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()