#!/usr/bin/env python3
"""
Descrição: Treino do modelo Baseline (One-Class SVM) focado no estado normal.
Isolamento absoluto: O Scaler nunca tem contacto com dados de anomalia.
"""

import os
import sys
import yaml
import pandas as pd
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value): return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)

PROCESSED_DATA_DIR = get_abs_path(config['paths']['processed_dir'])
MODELS_DIR = get_abs_path(config['paths']['models_dir'])

NU = config['models']['oc_svm']['nu']
GAMMA = config['models']['oc_svm']['gamma']
KERNEL = config['models']['oc_svm']['kernel']

def train_baseline():
    print("=== A iniciar Treino do Modelo de Baseline ===")
    data_path = os.path.join(PROCESSED_DATA_DIR, "ims_bearing1_features.csv")
    if not os.path.exists(data_path): return
        
    df = pd.read_csv(data_path)
    df_baseline = df[df["Scenario"] == "D0_Baseline"].copy()
    
    # Excluir metadados conhecidos e colunas de simulação para evitar furos de dimensionalidade
    excluir = ['Timestamp', 'Filename', 'Scenario', 'Scenario_Simulado']
    features_col = [c for c in df.columns if c not in excluir]
    
    X_train = df_baseline[features_col]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    ocsvm = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)
    ocsvm.fit(X_train_scaled)
    
    # Salvar artefactos puros criados estritamente na janela de treino histórica
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "baseline_scaler.pkl"))
    joblib.dump(ocsvm, os.path.join(MODELS_DIR, "baseline_ocsvm.pkl"))
    print("=== Treino Concluído com Sucesso Sem Data Leakage ===")

if __name__ == "__main__":
    train_baseline()