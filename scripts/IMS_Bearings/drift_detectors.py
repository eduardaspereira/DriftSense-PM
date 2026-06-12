#!/usr/bin/env python3
"""
Descrição: Implementação e avaliação dos detetores de Concept Drift (Det1 e Det2).
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp

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

# Parametros dos detetores extraídos do YAML
WINDOW_SIZE = config['detectors']['det2_distribution_test']['window_size_ks']
KS_P_VALUE_TH = config['detectors']['det2_distribution_test']['alpha_ks']
PERSISTENCE_TH = config['detectors']['det1_error_monitoring']['persistence']

def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "baseline_scaler.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "baseline_ocsvm.pkl"))
    return scaler, model

def run_evaluation():
    print("=== A iniciar Avaliação de Concept Drift ===")
    
    data_path = os.path.join(PROCESSED_DATA_DIR, "ims_bearing1_features.csv")
    df = pd.read_csv(data_path)
    scaler, model = load_artifacts()
    
    # Isolar as features dinamicamente excluindo as colunas textuais e de metadados
    features_col = [c for c in df.columns if c not in ['Timestamp', 'Filename', 'Scenario']]
    
    # Extrair os dados de referência (primeira janela do D0) para o Det2
    reference_data = df[df["Scenario"] == "D0_Baseline"].iloc[:100]
    ref_feature = reference_data["Energy_HighFreq"].values
    
    det1_alarms = []
    det2_alarms = []
    
    print("A simular inferência sequencial na Edge...")
    
    # 2. Varrer o dataset simulando a chegada de dados ao longo do tempo
    for i in range(len(df) - WINDOW_SIZE):
        janela_atual = df.iloc[i : i + WINDOW_SIZE]
        cenario_atual = janela_atual.iloc[-1]["Scenario"]
        
        # --- AVALIAÇÃO DET1 (Error Monitoring via Persistência Secuencial) ---
        X_janela = janela_atual[features_col]
        X_scaled = scaler.transform(X_janela)
        
        # Previsão: 1 (Normal), -1 (Anomalia)
        preds = model.predict(X_scaled)
        
        # Conta a quantidade de erros seguidos no fim da janela para cumprir a persistência
        erros_consecutivos = 0
        for p in reversed(preds):
            if p == -1:
                erros_consecutivos += 1
            else:
                break
        
        if erros_consecutivos >= PERSISTENCE_TH:
            det1_alarms.append({
                "index": i + WINDOW_SIZE, 
                "scenario": cenario_atual, 
                "consecutive_errors": erros_consecutivos
            })
            
        # --- AVALIAÇÃO DET2 (Statistical - Kolmogorov-Smirnov) ---
        current_feature = janela_atual["Energy_HighFreq"].values
        
        # O teste devolve estatística e p-value.
        stat, p_value = ks_2samp(ref_feature, current_feature)
        
        if p_value < KS_P_VALUE_TH:
            det2_alarms.append({
                "index": i + WINDOW_SIZE, 
                "scenario": cenario_atual, 
                "p_value": p_value
            })
            
    # 3. Analisar Resultados para o Artigo Científico
    print("\n--- Resultados DET1 (Performance/Error Monitoring) ---")
    d1_fps = [x for x in det1_alarms if x["scenario"] == "D0_Baseline"]
    d1_tps = [x for x in det1_alarms if x["scenario"] != "D0_Baseline"]
    print(f"Falsos Positivos no Baseline: {len(d1_fps)} janelas acionadas")
    if d1_tps:
        print(f"Detection Delay (ficheiro onde disparou no drift): Ficheiro nº {d1_tps[0]['index']}")
        
    print("\n--- Resultados DET2 (Distribution / Covariate Shift) ---")
    d2_fps = [x for x in det2_alarms if x["scenario"] == "D0_Baseline"]
    d2_tps = [x for x in det2_alarms if x["scenario"] != "D0_Baseline"]
    print(f"Falsos Positivos no Baseline: {len(d2_fps)} janelas acionadas")
    if d2_tps:
        print(f"Detection Delay (ficheiro onde disparou no drift): Ficheiro nº {d2_tps[0]['index']}")

if __name__ == "__main__":
    run_evaluation()