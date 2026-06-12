#!/usr/bin/env python3
"""
Descrição: Treino do modelo Baseline (One-Class SVM) focado no estado normal.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
"""

import os
import sys
import yaml
import pandas as pd
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# --- CARREGAR CONFIGURAÇÃO DINAMICAMENTE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERRO: Ficheiro de configuração não encontrado em: {CONFIG_PATH}")
    sys.exit(1)

PROCESSED_DATA_DIR = get_abs_path(config['paths']['processed_dir'])
MODELS_DIR = get_abs_path(config['paths']['models_dir'])

# Hiperparâmetros baseados no ficheiro YAML
NU = config['models']['oc_svm']['nu']
GAMMA = config['models']['oc_svm']['gamma']
KERNEL = config['models']['oc_svm']['kernel']

def train_baseline():
    print("=== A iniciar Treino do Modelo de Baseline ===")
    
    # 1. Carregar o dataset processado
    data_path = os.path.join(PROCESSED_DATA_DIR, "ims_bearing1_features.csv")
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado em {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Isolar apenas os dados de treino (D0_Baseline)
    df_baseline = df[df["Scenario"] == "D0_Baseline"].copy()
    print(f"Amostras de treino (D0_Baseline): {len(df_baseline)}")
    
    # Selecionar as features a usar pelo modelo (excluir metadados dinamicamente)
    features_col = [c for c in df.columns if c not in ['Timestamp', 'Filename', 'Scenario']]
    
    X_train = df_baseline[features_col]
    
    # 3. Escalar as features (Obrigatório para SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Treinar a One-Class SVM com os parâmetros do YAML
    ocsvm = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)
    ocsvm.fit(X_train_scaled)
    
    # Testar no próprio set de treino para garantir que não há overfit gigante
    preds = ocsvm.predict(X_train_scaled)
    falsos_alarmes = (preds == -1).sum() # -1 indica anomalia
    fpr = falsos_alarmes / len(preds)
    print(f"Taxa de Falsos Alarmes (FPR) no treino: {fpr:.4f}")
    
    # 5. Guardar modelo e scaler para inferência na Edge
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "baseline_scaler.pkl"))
    joblib.dump(ocsvm, os.path.join(MODELS_DIR, "baseline_ocsvm.pkl"))
    
    print(f"Artefactos guardados em: {MODELS_DIR}")
    print("=== Treino Concluído com Sucesso ===")

if __name__ == "__main__":
    train_baseline()