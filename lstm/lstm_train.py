import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, 'lstm_output')
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_ret.csv')

PARAMS_PATH = os.path.join(ROOT_DIR, 'lstm_best_params.json')
FEATURES_PATH = os.path.join(ROOT_DIR, 'lstm_selected_features.json')
SCALER_X_PATH = os.path.join(ROOT_DIR, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(ROOT_DIR, 'scaler_y.pkl')
MODEL_PATH = os.path.join(ROOT_DIR, 'training_results', 'best_lstm_model.keras')
PLOT_PATH = os.path.join(ROOT_DIR, 'training_results', 'learning_curve.png')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def build_model_from_config(config, n_features):
    st = config['structure']
    lc = config['layers_config']
    
    model = Sequential()
    # SEQ_LEN z JSON
    model.add(Input(shape=(st['seq_len'], n_features)))
    
    # Budowa warstw według przepisu
    for layer_conf in lc:
        model.add(LSTM(
            units=layer_conf['units'], 
            return_sequences=layer_conf['return_sequences']
        ))
        if layer_conf['dropout'] > 0:
            model.add(Dropout(layer_conf['dropout']))
            
    if st['use_dense']:
        model.add(Dense(st['dense_units'], activation=st['dense_activation']))
        
    model.add(Dense(1))
    
    # Kompilacja
    lr = st['learning_rate']
    if st['optimizer'] == 'adam': opt = Adam(learning_rate=lr)
    elif st['optimizer'] == 'rmsprop': opt = RMSprop(learning_rate=lr)
    else: opt = SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    print("--- TRENING LSTM ---")
    
    # 1. Wczytanie konfiguracji
    with open(PARAMS_PATH) as f: config = json.load(f)
    with open(FEATURES_PATH) as f: features = json.load(f)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    
    # Parametry z JSON
    SEQ_LEN = config['structure']['seq_len']
    BATCH_SIZE = config['structure']['batch_size']
    SPLIT_DATE = config['meta']['split']
    TARGET_COL = config['meta']['target']
    
    print(f"Parametry z JSON: SEQ={SEQ_LEN}, BATCH={BATCH_SIZE}, Cechy={len(features)}")
    
    # 2. Dane
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    
    df_train = df.loc[df.index < SPLIT_DATE].copy()
    X_train = df_train[features]
    y_train = df_train[[TARGET_COL]]
    
    X_sc = scaler_X.transform(X_train)
    y_sc = scaler_y.transform(y_train)

    X_seq, y_seq = create_sequences(X_sc, y_sc, SEQ_LEN)
    
    # 3. Trening
    model = build_model_from_config(config, len(features))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]
    
    history = model.fit(
        X_seq, y_seq,
        epochs=150,
        batch_size=BATCH_SIZE, # Używamy BATCH_SIZE z JSON
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Wykres
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_PATH)
    print(f"Zakończono. Model: {MODEL_PATH}")