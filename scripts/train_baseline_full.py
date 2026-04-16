import os
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 1. CONFIGURAÇÕES E PASTAS (Selo de Reprodutibilidade ACM) [cite: 511, 512]
CONFIG_PATH = "../configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config['paths']['processed_dir']
FIGURES_DIR = config['paths']['figures_dir']
METRICS_DIR = config['paths']['results_dir']
MODELS_DIR = config['paths']['models_dir']

for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

target_names = ['Anomalia/Drift (-1)', 'Normal (1)']

# 2. CARREGAMENTO E SPLIT (D0 - 100% Normal) [cite: 16, 214]
print("📂 A carregar dados e preparar o Benchmark...")
caminho_d0 = os.path.join(PROCESSED_DIR, "D0_dataset_features.csv")
df_d0 = pd.read_csv(caminho_d0)

X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
y_d0 = np.ones(len(X_d0)) # 1 para Normal

# Split cronológico (80% treino / 20% teste normal) para evitar leakage [cite: 11]
X_train, X_test_normal, y_train, y_test_normal = train_test_split(X_d0, y_d0, test_size=0.2, shuffle=False)

# 3. PREPARAR DADOS DE DRIFT (D1, D3, D4) PARA TESTE
test_anomalies = []
for file in os.listdir(PROCESSED_DIR):
    if file.endswith(".csv") and not file.startswith("D0"):
        df_anom = pd.read_csv(os.path.join(PROCESSED_DIR, file))
        X_anom = df_anom.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
        y_anom = np.full(len(X_anom), -1) # -1 para Anomalia
        test_anomalies.append((X_anom, y_anom))

X_test = pd.concat([X_test_normal] + [anom[0] for anom in test_anomalies], ignore_index=True)
y_test = np.concatenate([y_test_normal] + [anom[1] for anom in test_anomalies])

# 4. NORMALIZAÇÃO (VITAL PARA SVM/LOF)
print("⚖️ A normalizar features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. DEFINIÇÃO E LOOP DE TREINO
models = {
    "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.001, random_state=42),
    "One-Class SVM": OneClassSVM(nu=0.01, kernel="rbf", gamma='scale'),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
}

print(f"🔬 A iniciar Benchmark em {len(X_test)} janelas de teste...")

for name, model in models.items():
    print(f"\n# MODELO: {name}")
    model.fit(X_train_scaled)
    y_pred = model.predict(X_test_scaled)
    
    # Gerar Report [cite: 163]
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    print(report_str)
    
    with open(os.path.join(METRICS_DIR, f"report_{name.replace(' ', '_').lower()}.txt"), "w") as f:
        f.write(f"Modelo: {name}\n{report_str}")
    
    # Matriz de Confusão 
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão: {name}')
    plt.savefig(os.path.join(FIGURES_DIR, f"cm_{name.replace(' ', '_').lower()}.png"))
    plt.close()

    # 6. EXPORTAR O VENCEDOR (LOF) E O SCALER [cite: 168, 512]
    if name == "Local Outlier Factor":
        joblib.dump(model, os.path.join(MODELS_DIR, 'baseline_model.pkl'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
        print(f"💾 VENCEDOR EXPORTADO: Modelo e Scaler guardados em {MODELS_DIR}")

print("\n🚀 TUDO PRONTO! Agora podes correr o 'run_detectors.py'.")