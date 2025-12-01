import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'raw_data/mwig40_d.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'plots/mwig40_alltime.png')

TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12

try:
    df = pd.read_csv(DATA_PATH, parse_dates=['Data'])
    df = df.sort_values('Data')

    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(df['Data'], df['Otwarcie'], label='Cena otwarcia')

    plt.title('Wykres kursu otwarcia (mWIG40)', fontsize = TITLE_FONT_SIZE)
    plt.xlabel('Data', fontsize = AXIS_FONT_SIZE)
    plt.ylabel('Wartość kursu otwarcia', fontsize = AXIS_FONT_SIZE)
    plt.xticks(fontsize = TICK_FONT_SIZE)
    plt.yticks(fontsize = TICK_FONT_SIZE)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.show()

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku pod ścieżką: {DATA_PATH}")
    print("Sprawdź, czy ścieżka do pliku jest poprawna.")
    sys.exit(1)
except KeyError:
    print("BŁĄD: W pliku CSV brakuje wymaganych kolumn ('Data' lub 'Zamkniecie').")
    sys.exit(1)
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")
    sys.exit(1)