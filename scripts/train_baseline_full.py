import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import warnings

warnings.filterwarnings('ignore')

# 1. Configurações e Pastas
CONFIG_PATH = "../configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config['paths']['processed_dir']
FIGURES_DIR = config['paths']['figures_dir']
METRICS_DIR = config['paths']['results_dir']

for folder in [FIGURES_DIR, METRICS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Convenção Scikit-Learn para Anomalias: 1 = Normal, -1 = Falha/Drift
target_names = ['Anomalia/Drift (-1)', 'Normal (1)']

# 2. Carregamento de Dados (Treino apenas com D0)
print("📂 A carregar dados e preparar o Benchmark...")

# CORREÇÃO: Usar exatamente o nome do ficheiro que está na tua pasta
caminho_d0 = os.path.join(PROCESSED_DIR, "D0_dataset_features.csv")
df_d0 = pd.read_csv(caminho_d0)

# Remover colunas não preditivas
X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
y_d0 = np.ones(len(X_d0)) # 1 para Normal

# Split cronológico do D0 (80% para ensinar a norma, 20% para testar a norma)
X_train, X_test_normal, y_train, y_test_normal = train_test_split(X_d0, y_d0, test_size=0.2, shuffle=False)

# 3. Carregar os Drifts EXCLUSIVAMENTE para o conjunto de Teste
test_anomalies = []
for file in os.listdir(PROCESSED_DIR):
    # CORREÇÃO: Apanha todos os ficheiros CSV que NÃO sejam o D0
    if file.endswith(".csv") and not file.startswith("D0"):
        df_anom = pd.read_csv(os.path.join(PROCESSED_DIR, file))
        X_anom = df_anom.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
        y_anom = np.full(len(X_anom), -1) # -1 para Anomalia/Drift
        
        # Junta aos dados de teste
        test_anomalies.append((X_anom, y_anom))

# Concatenar todos os dados de teste (20% do Normal + 100% dos Drifts)
X_test = pd.concat([X_test_normal] + [anom[0] for anom in test_anomalies], ignore_index=True)
y_test = np.concatenate([y_test_normal] + [anom[1] for anom in test_anomalies])

print(f"🔬 Divisão Completa:")
print(f"   -> Treino (100% Normal): {len(X_train)} janelas")
print(f"   -> Teste (Normais + Drifts): {len(X_test)} janelas")

# 4. Definição dos Modelos Não Supervisionados
models = {
    "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.01, random_state=42),
    "One-Class SVM": OneClassSVM(nu=0.01, kernel="rbf", gamma='scale'),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
}

# 5. Loop de Treino e Avaliação
for name, model in models.items():
    print(f"\n{'='*50}\n# MODELO: {name}\n{'='*40}")
    
    # Treino apenas nos dados normais
    if name == "Local Outlier Factor":
        model.fit(X_train) # LOF precisa do novelty=True para fazer predict depois
    else:
        model.fit(X_train)
        
    # Previsão sobre o conjunto de teste (Normais + Falhas)
    y_pred = model.predict(X_test)
    
    # A) CLASSIFICATION REPORT
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    print("📋 Classification Report:")
    print(report_str)
    
    # Guardar Report
    report_path = os.path.join(METRICS_DIR, f"report_{name.replace(' ', '_').lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"Modelo: {name}\n{report_str}")
    
    # B) MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão: {name}\n(Treino 100% Normal vs Teste com Drifts)')
    plt.ylabel('Classe Real (Física)')
    plt.xlabel('Previsão do Modelo')
    
    fig_path = os.path.join(FIGURES_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fig_path)
    plt.close()

print("\n🚀 BENCHMARK CONCLUÍDO! Imagens e métricas geradas com sucesso.")