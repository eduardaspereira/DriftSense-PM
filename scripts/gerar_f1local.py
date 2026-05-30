import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report

# 1. Configurações Visuais
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'figure.dpi': 300})

PROCESSED_DIR = "../data/processed/" # Ajusta para a tua pasta
MODELS_DIR = "../models/"            # Ajusta para a tua pasta

# 2. Carregar o Modelo e o Scaler
print("📦 A carregar o modelo base e o normalizador...")
model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# 3. Ordem cronológica dos cenários
cenarios = ['D0', 'D1', 'D2', 'D3', 'D4'] # Adiciona os de transição se quiseres

historico = []

print("🚀 A simular previsões locais para obter o F1-Score (Cenário A0)...")
for cenario in cenarios:
    ficheiro = [f for f in os.listdir(PROCESSED_DIR) if f.startswith(cenario) and f.endswith('.csv')]
    if not ficheiro:
        continue
    
    df = pd.read_csv(os.path.join(PROCESSED_DIR, ficheiro[0]))
    features = [c for c in df.columns if c not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
    
    # Prever todas as janelas do cenário atual de uma vez (super rápido no PC)
    X_scaled = scaler.transform(df[features])
    previsoes = model.predict(X_scaled)
    
    # 1 para D0 (Normal), -1 para os restantes (Anomalia/Drift)
    true_label = 1 if cenario == 'D0' else -1
    
    for i in range(len(previsoes)):
        historico.append({
            'Cenario': cenario,
            'True_Label': true_label,
            'Predicted_Label': previsoes[i]
        })

df_a0 = pd.DataFrame(historico)

# 4. Calcular Métricas Globais
print("\n📊 RELATÓRIO DO MODELO SEM ADAPTAÇÃO (A0)")
print(classification_report(df_a0['True_Label'], df_a0['Predicted_Label'], digits=4))

# 5. Calcular F1-Score Contínuo (Rolling Window de 50 janelas)
window_size = 50
rolling_f1 = []
for i in range(len(df_a0)):
    inicio = max(0, i - window_size)
    y_true = df_a0['True_Label'].iloc[inicio:i+1]
    y_pred = df_a0['Predicted_Label'].iloc[inicio:i+1]
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    rolling_f1.append(f1)

df_a0['F1_Rolling'] = rolling_f1

# 6. Gerar Gráfico de Degradação
plt.figure(figsize=(10, 4))
sns.lineplot(x=df_a0.index, y=df_a0['F1_Rolling'], color='#e74c3c', linewidth=2, label='A0 (Sem Adaptação)')
plt.title('Colapso do F1-Score perante Concept Drift (Sem Adaptação)')
plt.xlabel('Janelas Temporais')
plt.ylabel('F1-Score (Rolling)')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('grafico_f1_colapso_a0.png')
print("✅ Gráfico guardado: grafico_f1_colapso_a0.png")