import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Configurações Visuais Académicas (Estilo Tese)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
FIGURES_DIR = "../results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 2. Carregar os Dados
df = pd.read_csv("../results/metrics/full_factorial_results.csv")

# Substituir "N/D" e "Não Recuperou" por NaN para a matemática funcionar
df['Delay (Janelas)'] = pd.to_numeric(df['Delay (Janelas)'], errors='coerce')
df['Recovery Time'] = pd.to_numeric(df['Recovery Time'], errors='coerce')
df['Latency (ms)'] = pd.to_numeric(df['Latency (ms)'], errors='coerce')

print("🎨 A gerar gráficos com qualidade de publicação...")

# =========================================================
# GRÁFICO 1: Detection Delay (DET1 vs DET2) - Sem Falsos Positivos
# =========================================================
plt.figure(figsize=(10, 6))
# Filtramos o D0 (baseline) e o DET0 (cego) para ver apenas as reações reais
df_detect = df[(df['Scenario'] != 'D0') & (df['Detector'] != 'DET0') & (df['Adaptation'] == 'A0')]

ax1 = sns.barplot(
    data=df_detect, 
    x='Scenario', 
    y='Delay (Janelas)', 
    hue='Detector', 
    palette=['#2ca02c', '#d62728'], # Verde e Vermelho
    edgecolor='black'
)
plt.title('Performance de Deteção: Atraso por Cenário de Falha', fontweight='bold', pad=15)
plt.ylabel('Atraso na Deteção (Número de Janelas)')
plt.xlabel('Cenário de Drift Injetado')
plt.legend(title='Estratégia de Deteção', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_detection_delay.png'), dpi=300)
plt.close()

# =========================================================
# GRÁFICO 2: Custo Computacional / Latência (A1 vs A2)
# =========================================================
plt.figure(figsize=(8, 6))
# Filtramos apenas as estratégias ativas (A1 e A2) e usamos os dados do melhor detetor (DET1)
df_latency = df[(df['Adaptation'].isin(['A1', 'A2'])) & (df['Detector'] == 'DET1') & (df['Scenario'] != 'D0')]

ax2 = sns.barplot(
    data=df_latency, 
    x='Adaptation', 
    y='Latency (ms)', 
    palette=['#1f77b4', '#ff7f0e'], # Azul e Laranja
    edgecolor='black',
    ci=None # Desliga as barras de erro para ficar mais limpo
)
plt.title('Custo Computacional na Edge: A1 vs A2', fontweight='bold', pad=15)
plt.ylabel('Latência de Adaptação (ms)')
plt.xlabel('Estratégia de Adaptação')

# Adicionar os valores exatos no topo das barras
for p in ax2.patches:
    ax2.annotate(format(p.get_height(), '.1f') + ' ms', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', 
                 xytext = (0, 9), 
                 textcoords = 'offset points',
                 fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_latency_comparison.png'), dpi=300)
plt.close()

print(f"✅ Gráficos guardados com sucesso em: {FIGURES_DIR}")
print("   - fig1_detection_delay.png")
print("   - fig2_latency_comparison.png")