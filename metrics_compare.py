import pandas as pd
import os

# Definicja ścieżki katalogu głównego (tam, gdzie znajduje się ten skrypt)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Słownik ścieżek zbudowany na podstawie Twoich skryptów
METRICS_FILES = {
    'LSTM': os.path.join(SCRIPT_DIR, 'lstm', 'lstm_output', 'prediction_results', 'lstm_metrics.csv'),
    'CART': os.path.join(SCRIPT_DIR, 'cart', 'cart_ret_output_selected_features', 'cart_metrics.csv'),
    'XGBoost': os.path.join(SCRIPT_DIR, 'XGBoost', 'xgboost_output', 'xgboost_metrics.csv'),
    'Naive': os.path.join(SCRIPT_DIR, 'naive', 'naive_metrics.csv'),
    'ARIMA': os.path.join(SCRIPT_DIR, 'arima', 'arima_metrics.csv')
}

def combine_metrics():
    all_data = []
    
    print("--- ZBIERANIE METRYK (Ścieżki Absolutne) ---")
    
    for model_name, full_path in METRICS_FILES.items():
        if os.path.exists(full_path):
            try:
                # Wczytanie metryk (format: Metric;Value z przecinkiem jako decimal)
                df = pd.read_csv(full_path, sep=';', decimal=',')
                
                # Przestawienie tabeli (Metrics na kolumny)
                df_pivot = df.set_index('Metric').T
                df_pivot.index = [model_name]
                
                all_data.append(df_pivot)
                print(f"[OK] Wczytano: {full_path}")
            except Exception as e:
                print(f"[BŁĄD] Problem z plikiem {model_name}: {e}")
        else:
            print(f"[BRAK] Nie znaleziono: {full_path}")

    if all_data:
        # Połączenie wszystkich wyników
        summary_df = pd.concat(all_data, sort=False)
        
        # Zapis końcowy w katalogu głównym projektu
        output_path = os.path.join(SCRIPT_DIR, 'all_models_metrics_comparison.csv')
        summary_df.to_csv(output_path, sep=';', decimal=',')
        
        print(f"\n--- ZAPISANO: {output_path} ---")
        print(summary_df)
    else:
        print("\n[BŁĄD] Brak danych do połączenia. Uruchom najpierw skrypty predykcyjne.")

if __name__ == "__main__":
    combine_metrics()