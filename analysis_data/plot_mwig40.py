import pandas as pd
import matplotlib.pyplot as plt
import sys

file_path = '../raw_data/mwig40_d.csv'

try:
    df = pd.read_csv(file_path, parse_dates=['Data'])
    df = df.sort_values('Data')

    plt.figure(figsize=(12, 6))
    plt.plot(df['Data'], df['Otwarcie'], label='Cena otwarcia')
    plt.title('Wykres kursu otwarcia (mWIG40)')
    plt.xlabel('Data')
    plt.ylabel('Wartość kursu otwarcia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('wykresy/mwig40_alltime.png')
    plt.show()

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku pod ścieżką: {file_path}")
    print("Sprawdź, czy ścieżka do pliku jest poprawna.")
    sys.exit(1)
except KeyError:
    print("BŁĄD: W pliku CSV brakuje wymaganych kolumn ('Data' lub 'Zamkniecie').")
    sys.exit(1)
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")
    sys.exit(1)