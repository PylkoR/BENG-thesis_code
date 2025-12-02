import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Konfiguracja ścieżek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset.csv')
OUTPUT_CORR_MATRIX = os.path.join(SCRIPT_DIR, 'correlation_matrix.png')
OUTPUT_CORR_TARGET = os.path.join(SCRIPT_DIR, 'correlation_target.png')

# Konfiguracja wykresów
TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 14
TICK_FONT_SIZE = 12

# Wczytanie danych
df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
df_numeric = df.drop(columns=['Data'])

# Macierz korelacji
corr_matrix = df_numeric.corr()

# 1. Wykres heatmap
plt.figure(figsize=(24, 20))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix, 
    mask=mask, 
    cmap='coolwarm', 
    center=0, 
    square=True, 
    linewidths=.5
)

plt.title('Macierz korelacji', fontsize=TITLE_FONT_SIZE)
plt.xticks(rotation=90, fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout()
plt.savefig(OUTPUT_CORR_MATRIX)
plt.close()

# 2. Korelacja z celem
target = 'mwig40_Zamkniecie'

if target in corr_matrix.columns:
    plt.figure(figsize=(12, 12))
    
    # Sortowanie
    target_corr = corr_matrix[target].sort_values(ascending=True).drop(target)
    
    # Kolorowanie
    colors = ['red' if x > 0.7 else ('blue' if x < -0.3 else 'gray') for x in target_corr.values]
    
    # Wykres
    target_corr.plot(kind='barh', color=colors, edgecolor='black')
    
    plt.title(f'Korelacja z {target}', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Współczynnik korelacji', fontsize=AXIS_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_CORR_TARGET)
    plt.close()