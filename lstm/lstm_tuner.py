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

# --- KONFIGURACJA ŚRODOWISKA ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_lstm.csv')
TUNER_DIR = os.path.join(SCRIPT_DIR, 'lstm_output', 'tuning_cache_dynamic')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'lstm_output')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'lstm_best_params.json')
SCALER_X_PATH = os.path.join(OUTPUT_DIR, 'scaler_x.pkl')
SCALER_Y_PATH = os.path.join(OUTPUT_DIR, 'scaler_y.pkl')

os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- KONFIGURACJA MODELU ---
TARGET_COL = 'mWIG40_Ret'
SPLIT_DATE = '2022-01-03'
VAL_SPLIT_RATIO = 0.2
MAX_EPOCHS = 30  

# Zmienna globalna na liczbę cech (będzie ustawiona w main)
n_features_global = None

# --- PRZETWARZANIE DANYCH ---
def create_sequences(X_data, y_data, seq_len):
    """Generowanie sekwencji czasowych o zadanej długości."""
    X, y = [], []
    if len(X_data) <= seq_len:
        return np.array(X), np.array(y)
    
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i:(i + seq_len)])
        y.append(y_data[i + seq_len])
    return np.array(X), np.array(y)

# --- DEFINICJA MODELU ---
def build_model(hp):
    model = Sequential()
    
    # POPRAWKA: Definiujemy parametry tutaj, zamiast używać hp.get()
    # Dzięki temu Tuner wie o nich przy inicjalizacji
    seq_len = hp.Int('seq_len', 30, 90, step=10)
    
    # Używamy zmiennej globalnej dla n_features lub hp.Fixed
    # hp.Fixed jest bezpieczniejsze, bo zapisuje wartość w logach tunera
    n_features = hp.Fixed('n_features', n_features_global)
    
    model.add(Input(shape=(seq_len, n_features)))
    
    # Warstwy LSTM
    num_lstm_layers = hp.Int('num_lstm_layers', 1, 3)
    
    for i in range(num_lstm_layers):
        units = hp.Int(f'lstm_units_{i}', 32, 256, step=32)
        dropout = hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)
        return_seq = True if i < num_lstm_layers - 1 else False
        
        model.add(LSTM(units=units, return_sequences=return_seq))
        if dropout > 0:
            model.add(Dropout(dropout))
    
    # Warstwy gęste
    if hp.Boolean('use_dense_layer'):
        dense_units = hp.Int('dense_units', 16, 128, step=16)
        activation = hp.Choice('dense_activation', ['relu', 'tanh', 'swish'])
        model.add(Dense(dense_units, activation=activation))
    
    model.add(Dense(1)) 

    # Kompilacja
    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    if optimizer_name == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer_name == 'rmsprop':
        opt = RMSprop(learning_rate=lr)
    else:
        opt = SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    return model

# --- NIESTANDARDOWY TUNER ---
class DynamicSeqTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # 1. Definicja hiperparametrów przestrzennych dla danych
        hp = trial.hyperparameters
        
        # Musimy zdefiniować seq_len tak samo jak w build_model
        seq_len = hp.Int('seq_len', 30, 90, step=10)
        
        # Rejestracja n_features dla spójności
        hp.Fixed('n_features', n_features_global)
        
        # 2. Generowanie danych dla konkretnej próby
        # Wykorzystuje zmienne globalne X_scaled_global, y_scaled_global
        X_seq, y_seq = create_sequences(X_scaled_global, y_scaled_global, seq_len)
        
        # Obsługa przypadku zbyt krótkich danych
        if len(X_seq) == 0:
            print(f"Pominięto próbę: seq_len={seq_len} zbyt duże dla zbioru danych.")
            return

        # 3. Uruchomienie standardowego procesu treningu z nowymi danymi
        batch_size = hp.Int('batch_size', 16, 64, step=16)
        
        return super().run_trial(
            trial, 
            *args, 
            x=X_seq, 
            y=y_seq, 
            batch_size=batch_size,
            **kwargs
        )

# --- WYKONANIE ---
if __name__ == "__main__":
    try:
        # 1. Import i czyszczenie
        print(f"Źródło: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',', index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()

        feature_cols = [c for c in df.columns if c != TARGET_COL]
        # Ustawienie zmiennej globalnej przed uruchomieniem Tunera
        n_features_global = len(feature_cols)
        
        # 2. Podział (Train only for Tuner)
        mask_train = df.index < SPLIT_DATE
        df_train = df.loc[mask_train].copy()
        
        print(f"Trening (do {SPLIT_DATE}): {len(df_train)} wierszy")

        # 3. Skalowanie (Fit na Train)
        scaler_X = MinMaxScaler((-1, 1))
        scaler_y = MinMaxScaler((-1, 1))
        
        X_scaled_global = scaler_X.fit_transform(df_train[feature_cols])
        y_scaled_global = scaler_y.fit_transform(df_train[[TARGET_COL]])
        
        # Zapis skalerów
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_y, SCALER_Y_PATH)

        # 4. Inicjalizacja Tunera
        print("Inicjalizacja DynamicSeqTuner (Hyperband)...")
        # Ważne: usuń stary katalog cache, jeśli zmieniałeś strukturę parametrów
        # overwrite=True powinno załatwić sprawę
        tuner = DynamicSeqTuner(
            build_model,
            objective='val_loss',
            max_epochs=MAX_EPOCHS,
            factor=3,
            directory=TUNER_DIR,
            project_name='mwig40_lstm_dynamic_v2', # Zmiana nazwy projektu dla pewności
            overwrite=True
        )
        
        tuner.search_space_summary()

        # 5. Uruchomienie szukania
        stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("Start optymalizacji...")
        tuner.search(
            validation_split=VAL_SPLIT_RATIO,
            callbacks=[stop_early],
            verbose=1
        )

        # 6. Eksport wyników
        best_hps = tuner.get_best_hyperparameters()[0]
        
        best_params = {
            'meta': {
                'target_col': TARGET_COL,
                'split_date': SPLIT_DATE,
                'features': feature_cols
            },
            'optimal_structure': {
                'seq_len': best_hps.get('seq_len'),
                'batch_size': best_hps.get('batch_size'),
                'num_lstm_layers': best_hps.get('num_lstm_layers'),
                'dense_units': best_hps.get('dense_units') if best_hps.get('use_dense_layer') else 0,
                'optimizer': best_hps.get('optimizer'),
                'learning_rate': best_hps.get('learning_rate')
            },
            'all_params': best_hps.values
        }
        
        # Szczegóły warstw
        best_params['layers_config'] = []
        for i in range(best_hps.get('num_lstm_layers')):
            best_params['layers_config'].append({
                'units': best_hps.get(f'lstm_units_{i}'),
                'dropout': best_hps.get(f'dropout_{i}')
            })

        with open(PARAMS_PATH, 'w') as f:
            json.dump(best_params, f, indent=4)
            
        print(f"\nZapisano konfigurację: {PARAMS_PATH}")
        print(f"Optymalne SEQ_LEN: {best_hps.get('seq_len')}")

    except Exception as e:
        print(f"Błąd wykonania: {e}")
        import traceback
        traceback.print_exc()