import pandas as pd
import os

# --- KONFIGURACJA ---
FILES_CONFIG = {
    'mwig40_d.csv': 'mwig40',  # Tabela bazowa
    '^dax_d.csv': 'dax',
    '^nkx_d.csv': 'nkx',
    '^spx_d.csv': 'spx',
    '^vix_d.csv': 'vix',
    '^wig20_d.csv': 'wig20',
    'brent.csv': 'brent',
    'eurpln_d.csv': 'eurpln',
    'pmi_d.csv': 'pmi',
    'usdpln_d.csv': 'usdpln'
}

# Ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'dataset.csv')

def load_and_prep(filename, prefix):
    """Wczytuje plik, parsuje daty, czyści specyficzne kolumny i dodaje prefiksy."""
    path = os.path.join(DATA_DIR, filename)
    try:
        df = pd.read_csv(path)
        
        # 1. Konwersja Daty
        df['Data'] = pd.to_datetime(df['Data'])
        
        # 2. Czyszczenie specyficzne dla ropy Brent (usuwanie znaków %)
        if 'brent' in filename:
            for col in ['Zmiana_ot', 'Zmiana_za']:
                if col in df.columns and df[col].dtype == 'object':
                     df[col] = df[col].str.replace('%', '').astype(float)
        
        # 3. Zmiana nazw kolumn (dodanie prefiksu)
        new_columns = {}
        for col in df.columns:
            if col != 'Data':
                new_columns[col] = f"{prefix}_{col}"
        df.rename(columns=new_columns, inplace=True)
        
        return df
    except Exception as e:
        print(f"[BŁĄD] Nie udało się przetworzyć {filename}: {e}")
        return None

def main():
    print("--- ROZPOCZYNAM PRZETWARZANIE DANYCH ---")
    
    # 1. Wczytanie ramki bazowej
    base_filename = 'mwig40_d.csv'
    base_prefix = FILES_CONFIG[base_filename]
    
    print(f"Wczytywanie bazy: {base_filename}...")
    df_main = load_and_prep(base_filename, base_prefix)
    
    if df_main is None:
        return

    # Sortowanie bazy po dacie
    df_main.sort_values('Data', inplace=True)

    # 2. Pętla łączenia (Left Join)
    for filename, prefix in FILES_CONFIG.items():
        if filename == base_filename:
            continue 
            
        print(f"Łączenie z: {filename}...")
        df_temp = load_and_prep(filename, prefix)
        
        if df_temp is not None:
            df_main = pd.merge(df_main, df_temp, on='Data', how='left')
            # Uzupełnianie luk w środku danych (np. święta) wartością poprzednią
            df_main = df_main.ffill()

    print(f"\nWymiary przed czyszczeniem: {df_main.shape}")

    # --- 3. CZYSZCZENIE ZER I BRAKÓW (NOWA SEKCJA) ---
    print("\n--- CZYSZCZENIE DANYCH ---")

    # A. Usuwanie kolumn zawierających zera (głównie uszkodzone Wolumeny)
    cols_with_zeros = [col for col in df_main.columns if (df_main[col] == 0).any()]
    
    if cols_with_zeros:
        print(f"Usuwam {len(cols_with_zeros)} kolumn zawierających zera (prawdopodobne błędy danych):")
        for col in cols_with_zeros:
            print(f" - {col}")
        df_main.drop(columns=cols_with_zeros, inplace=True)
    else:
        print("Nie znaleziono kolumn z zerami.")

    # B. Usuwanie wierszy początkowych z brakami (NaN)
    # Po merge i ffill, na początku tabeli mogą być NaNy, jeśli inne indeksy zaczynają się później niż mwig40
    rows_before = len(df_main)
    df_main.dropna(inplace=True)
    rows_after = len(df_main)
    
    if rows_before != rows_after:
        print(f"\nPrzycięto dane do wspólnego zakresu dat.")
        print(f"Usunięto {rows_before - rows_after} początkowych wierszy.")
        print(f"Nowy zakres dat: od {df_main['Data'].min().date()} do {df_main['Data'].max().date()}")
    
    # 4. Zapis do CSV
    print(f"\nZapisywanie do pliku: {OUTPUT_FILE}")
    
    # Format pod polskiego Excela/CSV
    df_main.to_csv(OUTPUT_FILE, index=False, sep=';', decimal=',')
    
    print("GOTOWE. Plik dataset.csv został utworzony.")
    print("-" * 30)
    print("Podgląd pierwszych 5 wierszy:")
    print(df_main.head())

if __name__ == "__main__":
    main()