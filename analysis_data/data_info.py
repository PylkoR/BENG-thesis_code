import pandas as pd
import os

# --- KONFIGURACJA ŚCIEŻEK (Metoda bezpieczna) ---

# 1. Pobierz pełną ścieżkę do katalogu, w którym leży TEN plik skryptu (czyli .../analysis/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Wyjdź piętro wyżej (do folderu projektu)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 3. Wejdź do folderu z danymi
DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')

# --- LISTA PLIKÓW ---
pliki = [
    'mwig40_d.csv', '^dax_d.csv', '^nkx_d.csv', '^spx_d.csv', '^vix_d.csv', 
    '^wig20_d.csv', 'brent.csv', 'eurpln_d.csv', 
    'pmi_d.csv', 'usdpln_d.csv'
]

dataframes = {}

print(f"Lokalizacja skryptu: {SCRIPT_DIR}")
print(f"Szukam danych w:     {DATA_DIR}\n")

# --- GŁÓWNA PĘTLA ---
for nazwa_pliku in pliki:
    sciezka_do_pliku = os.path.join(DATA_DIR, nazwa_pliku)
    
    try:
        # Wczytanie danych
        df = pd.read_csv(sciezka_do_pliku)
        dataframes[nazwa_pliku] = df
        
        print(f"PLIK: {nazwa_pliku}")
        print(f"Wymiary (wiersze, kolumny): {df.shape}")
        print(f"Kolumny: {df.columns.tolist()}")
        
        # 1. Analiza braków danych (NaN)
        braki = df.isnull().sum()
        braki_wystepujace = braki[braki > 0]
        
        if not braki_wystepujace.empty:
            print("Znaleziono braki danych (NaN) (kolumna: liczba braków):")
            print(braki_wystepujace)
        else:
            print("Status: Brak pustych wartości (NaN).")

        # 2. Analiza wartości zerowych (0)
        zera = (df == 0).sum()
        zera_wystepujace = zera[zera > 0]

        if not zera_wystepujace.empty:
            print("Znaleziono wartości ZERO (0) (kolumna: liczba zer):")
            print(zera_wystepujace)
        else:
            print("Status: Brak wartości zerowych.")
            
        print("-" * 100)
        
    except FileNotFoundError:
        print(f"[BŁĄD] Nie znaleziono pliku: {nazwa_pliku}")
    except Exception as e:
        print(f"[BŁĄD] Problem z plikiem {nazwa_pliku}: {e}")