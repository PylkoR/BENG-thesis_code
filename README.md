# Prognozowanie kursu indeksu mWIG40 różnymi metodami 📈

Repozytorium zawiera kod i dane użyte do porównania kilku metod prognozowania dla danych rynkowych (m.in. LSTM, ARIMA, CART oraz metoda naiwna). Projekt skupia się na przygotowaniu danych, trenowaniu modeli, tuningu hiperparametrów oraz zapisaniu wyników i metryk.

---

## Struktura repozytorium 🗂️

- `raw_data/` – surowe pliki CSV ze źródłowymi danymi rynkowymi (indeksy, surowce, kursy walut itp.).
- `analysis_data/` – skrypty do wstępnej analizy, łączenia i wizualizacji danych (np. wykresy, analizy korelacji) oraz analiza wyników prognoz w skrypcie `results_analysis.ipynb`
- `lstm/` – implementacja i eksperymenty z modelami LSTM: przygotowanie danych, skrypty treningowe i do predykcji, tunery (random i hyperband), oraz katalog `lstm_output/` z wynikami (metryki, predykcje, zapisane modele i cache tuningu).
- `cart/` – eksperymenty z regresją drzewiastą (CART) dla cen i zwrotów; zawiera skrypty treningowe oraz katalogi z wynikami i najlepszymi parametrami (`cart_*_output/`).
- `arima/` – skrypty związane z modelami ARIMA oraz pliki z metrykami i predykcjami.
- `naive/` – prosty benchmark: skrypt generujący prognozy metodą naiwną oraz wygenerowane wykresy i metryki (podstawowe porównanie z modelami bardziej zaawansowanymi).
- Pliki CSV na poziomie głównym (np. `dataset.csv`, `dataset_ret.csv`, `dataset_cart.csv`) – przygotowane zbiory danych używane w eksperymentach. Plik `dataset.csv` zawiera kompletne dane (oczyszczone i zsynchronizowane), jest bazą każdy inny plik z danymi jest jego pochodną. 
- `requirements.txt` – lista zależności potrzebnych do uruchomienia skryptów.
