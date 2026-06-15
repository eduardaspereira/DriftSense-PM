#!/usr/bin/env python3
"""
Descrição: Matriz Fatorial com Veto Dinâmico por Score e Profiling de Energia (mJ).
CORREÇÃO DE LEAKAGE: Scalers e limiares de densidade (percentil) são ajustados 
estritamente no set de treino passado, garantindo uma stream cega simulada.
"""

import os
import yaml
import json
import time
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)

FEATURES = config['feature_engineering']['features']
PROCESSED_DIR = config['paths']['processed_dir']
TARGET_RESULTS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "results/IMS_Bearings/metrics/2nd_test"))
os.makedirs(TARGET_RESULTS_DIR, exist_ok=True)

WINDOW_SIZE = 100
ALPHA_KS = 0.001
PERCENTAGEM_REPLAY = 0.15
POWER_IDLE_W = 2.7
POWER_MAX_W = 6.5

def estimar_energia_mj(tempo_ms):
    return (POWER_IDLE_W + (POWER_MAX_W - POWER_IDLE_W)) * tempo_ms

def executar_estrategia_por_ficheiro(csv_filename, label_column, cenario_tag):
    csv_path = os.path.normpath(os.path.join(PROJECT_ROOT, "data/IMS_Bearings/processed/2nd_test", csv_filename))
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    df["Energy_HighFreq"] = df["Energy_HighFreq"].ewm(alpha=0.2, adjust=False).mean()
    
    # Divisão estritamente baseada no histórico nominal passado
    limite_tr = int(len(df[df[label_column] == "D0_Baseline"]) * 0.8)
    df_treino = df.iloc[:limite_tr].copy()
    df_stream = df.iloc[limite_tr:].copy()
    y_stream = np.where(df_stream[label_column] == "D0_Baseline", 1, -1)
    
    indices_drift = np.where(y_stream == -1)[0]
    idx_X = indices_drift[0] if len(indices_drift) > 0 else None
    idx_Y = None
    
    X_tr_bruto = df_treino[FEATURES].values
    X_st_bruto = df_stream[FEATURES].values
    
    nomes_est = ["A0_None", "A1_Periodic", "A2_Lightweight_Veto"]
    modelos = {nome: {"scaler": StandardScaler(), "modelo": LocalOutlierFactor(novelty=True), "retrains": 0, "vetos": 0, "energia": 0.0} for nome in nomes_est}
    
    # LIMITAÇÃO DE ESCALA REALISTA: Ajuste inicial feito exclusivamente no passado estável
    X_init_scaled = modelos["A2_Lightweight_Veto"]["scaler"].fit_transform(X_tr_bruto)
    modelos["A2_Lightweight_Veto"]["modelo"].fit(X_init_scaled)
    ref_scores = modelos["A2_Lightweight_Veto"]["modelo"].score_samples(X_init_scaled)
    ref_score_th = np.percentile(ref_scores, 5) # Limiar de Veto cego extraído do histórico
    
    for nome, st in modelos.items():
        if nome != "A2_Lightweight_Veto":
            st["modelo"].fit(st["scaler"].fit_transform(X_tr_bruto))
        st["preds"] = []

    ref_dist = df_treino["Energy_HighFreq"].values[-WINDOW_SIZE:]
    
    for i in range(len(X_st_bruto)):
        x_atual = X_st_bruto[i].reshape(1, -1)
        
        for nome, st in modelos.items():
            # Cada modelo utiliza o seu transformador ajustado na linha de base histórica
            st["preds"].append(st["modelo"].predict(st["scaler"].transform(x_atual))[0])
            
        if i >= WINDOW_SIZE:
            janela_rec = X_st_bruto[i-WINDOW_SIZE:i]
            
            if i % 50 == 0:
                t0 = time.perf_counter()
                modelos["A1_Periodic"]["modelo"].fit(modelos["A1_Periodic"]["scaler"].fit_transform(janela_rec))
                modelos["A1_Periodic"]["retrains"] += 1
                modelos["A1_Periodic"]["energia"] += estimar_energia_mj((time.perf_counter()-t0)*1000)
                
            f_recente = df_stream.iloc[i-WINDOW_SIZE:i]["Energy_HighFreq"].values
            _, p_val = ks_2samp(ref_dist, f_recente)
            
            if p_val < ALPHA_KS:
                if idx_X is not None and i >= idx_X and idx_Y is None:
                    idx_Y = i
                
                x_jan_scaled = modelos["A2_Lightweight_Veto"]["scaler"].transform(janela_rec)
                scores_j = modelos["A2_Lightweight_Veto"]["modelo"].score_samples(x_jan_scaled)
                
                if np.mean(scores_j) < ref_score_th:
                    modelos["A2_Lightweight_Veto"]["vetos"] += 1
                    ref_dist = f_recente
                else:
                    t0 = time.perf_counter()
                    idx_rep = np.random.choice(len(X_tr_bruto), int(len(X_tr_bruto)*PERCENTAGEM_REPLAY), replace=False)
                    X_buf = np.vstack((X_tr_bruto[idx_rep], janela_rec))
                    modelos["A2_Lightweight_Veto"]["modelo"].fit(modelos["A2_Lightweight_Veto"]["scaler"].fit_transform(X_buf))
                    modelos["A2_Lightweight_Veto"]["retrains"] += 1
                    modelos["A2_Lightweight_Veto"]["energia"] += estimar_energia_mj((time.perf_counter()-t0)*1000)
                    ref_dist = f_recente

    delay_val = int(idx_Y - idx_X) if (idx_X is not None and idx_Y is not None) else None

    print(f"\n================ MATRIZ FATORIAL: {cenario_tag.upper()} ================")
    export_dict = {}
    for nome in nomes_est:
        f1 = f1_score(y_stream, modelos[nome]["preds"], average='macro')
        export_dict[nome] = {
            "f1_score": round(f1, 4), "retrains": modelos[nome]['retrains'], "vetos": modelos[nome]['vetos'],
            "energia_mj": round(modelos[nome]['energia'], 2), "detection_delay": delay_val if nome == "A2_Lightweight_Veto" else None
        }
    with open(os.path.join(TARGET_RESULTS_DIR, f"metrics_{cenario_tag}.json"), "w") as f:
        json.dump(export_dict, f, indent=4)

if __name__ == "__main__":
    executar_estrategia_por_ficheiro("simulacao_cenario1_regime.csv", "Scenario_Simulado", "cenario1")
    executar_estrategia_por_ficheiro("simulacao_cenario2_falha.csv", "Scenario_Simulado", "cenario2")