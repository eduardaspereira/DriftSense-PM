"""
Descrição: Treino do modelo baseline e benchmark de detetores.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Este script treina vários detectores (IsolationForest, One-Class SVM, LOF),
executa um benchmark sobre janelas de teste e exporta o modelo vencedor e
o scaler para uso posterior na avaliação factorial.
"""

import os
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_project_path(path_value):
    normalized = path_value.removeprefix("../")
    return os.path.normpath(os.path.join(PROJECT_ROOT, normalized))

parser = argparse.ArgumentParser(description="Treino offline dos 3 modelos de baseline.")
parser.add_argument("--if-n-estimators", type=int, default=100)
parser.add_argument("--if-contamination", type=float, default=0.1)
parser.add_argument("--svm-nu", type=float, default=0.01)
parser.add_argument("--svm-kernel", type=str, default="rbf")
parser.add_argument("--svm-gamma", type=str, default="scale")
parser.add_argument("--lof-n-neighbors", type=int, default=20)
parser.add_argument("--lof-contamination", type=float, default=1.0)
args = parser.parse_args()

# 1. CONFIGURAÇÕES E PASTAS (Selo de Reprodutibilidade ACM)
CONFIG_PATH = resolve_project_path("../configs/config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = resolve_project_path(config['paths']['processed_dir'])
FIGURES_DIR = resolve_project_path(config['paths']['figures_dir'])
METRICS_DIR = resolve_project_path(config['paths']['results_dir'])
MODELS_DIR = resolve_project_path(config['paths']['models_dir'])

for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

target_names = ['Anomalia/Drift (-1)', 'Normal (1)']

# 2. CARREGAMENTO E SPLIT (D0 - 100% Normal)
print("A carregar dados e preparar o Benchmark...")
caminho_d0 = os.path.join(PROCESSED_DIR, "D0_dataset_features.csv")
if not os.path.exists(caminho_d0):
    raise FileNotFoundError(f"Ficheiro de treino baseline não encontrado: {caminho_d0}")
df_d0 = pd.read_csv(caminho_d0)

X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
y_d0 = np.ones(len(X_d0)) # 1 para Normal

# Split cronológico (80% treino / 20% teste normal) para evitar leakage
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
print("A normalizar features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. DEFINIÇÃO E LOOP DE TREINO
models = {
    "Isolation Forest": IsolationForest(
        n_estimators=args.if_n_estimators,
        contamination=args.if_contamination / 100,
        random_state=42
    ),
    "One-Class SVM": OneClassSVM(
        nu=args.svm_nu,
        kernel=args.svm_kernel,
        gamma=args.svm_gamma
    ),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=args.lof_n_neighbors,
        contamination=args.lof_contamination / 100,
        novelty=True
    )
}

print(f"A iniciar Benchmark em {len(X_test)} janelas de teste...")

# 6. AVALIAR TODOS OS MODELOS E ESCOLHER O MELHOR
best_model = None
best_model_name = None
best_f1_score = -1
model_results = {}

for name, model in models.items():
    print(f"\n# MODELO: {name}")
    model.fit(X_train_scaled)
    y_pred = model.predict(X_test_scaled)
    
    # Gerar Report
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    print(report_str)
    
    # Extrair F1-score (weighted) para seleção
    lines = report_str.split('\n')
    for line in lines:
        if 'weighted avg' in line:
            parts = line.split()
            f1_weighted = float(parts[-2])  # F1 score está penúltima coluna
            model_results[name] = f1_weighted
            print(f"{name} F1 (weighted): {f1_weighted:.3f}")
            
            if f1_weighted > best_f1_score:
                best_f1_score = f1_weighted
                best_model = model
                best_model_name = name
            break
    
    with open(os.path.join(METRICS_DIR, f"report_{name.replace(' ', '_').lower()}.txt"), "w") as f:
        f.write(f"Modelo: {name}\n{report_str}")
    
    # Matriz de Confusão 
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão: {name}')
    plt.savefig(os.path.join(FIGURES_DIR, f"cm_{name.replace(' ', '_').lower()}.png"))
    plt.close()

# 7. EXPORTAR O VENCEDOR COM JUSTIFICAÇÃO 
print(f"\n{'='*60}")
print(f"VENCEDOR SELECIONADO: {best_model_name}")
print(f"   F1-Score (weighted): {best_f1_score:.3f}")
for name, f1 in model_results.items():
    status = "VENCEDOR" if name == best_model_name else f"(F1={f1:.3f})"
    print(f"   - {name}: {status}")
print(f"{'='*60}\n")

joblib.dump(best_model, os.path.join(MODELS_DIR, 'baseline_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print(f"VENCEDOR EXPORTADO: {best_model_name}")
print(f"   Modelo e Scaler guardados em {MODELS_DIR}")