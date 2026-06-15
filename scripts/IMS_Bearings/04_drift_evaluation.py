#!/usr/bin/env python3
"""
Descrição: Avaliação de Concept Drift na Camada Semântica.
Aplica os testes estatísticos (KS e PSI) sobre o fluxo de Anomaly Scores.
NOTA CIENTÍFICA: Para mitigar a violação da assunção i.i.d. induzida pela 
autocorrelação do filtro EMA, o script foi atualizado para efetuar a avaliação 
via sub-amostragem (downsampling) não sobreposta com passo igual a WINDOW_SIZE.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value): return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)

PROCESSED_DATA_DIR = get_abs_path(config['paths']['processed_dir'])
MODELS_DIR = get_abs_path(config['paths']['models_dir'])
WINDOW_SIZE = config['detectors']['det2_distribution_test']['window_size_ks']
KS_P_VALUE_TH = config['detectors']['det2_distribution_test']['alpha_ks']
PSI_THRESHOLD = 0.25

def calcular_psi(expected, actual, num_bins=10):
    counts_expected, bin_edges = np.histogram(expected, bins=num_bins, density=False)
    counts_actual, _ = np.histogram(actual, bins=bin_edges, density=False)
    expected_pct = (counts_expected + 0.5) / (len(expected) + 0.5 * num_bins)
    actual_pct = (counts_actual + 0.5) / (len(actual) + 0.5 * num_bins)
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "baseline_scaler.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "baseline_ocsvm.pkl"))
    return scaler, model

def run_score_based_evaluation(csv_name, label_col, nome_experiencia):
    print(f"\n--- A avaliar Camada Semântica: {nome_experiencia} ---")
    data_path = os.path.join(PROCESSED_DATA_DIR, csv_name)
    if not os.path.exists(data_path):
        print(f"[Aviso] Dataset {csv_name} em falta. Salto.")
        return
        
    df = pd.read_csv(data_path)
    scaler, model = load_artifacts()
    features_col = [c for c in df.columns if c not in ['Timestamp', 'Filename', 'Scenario', 'Scenario_Simulado']]
    
    # 1. Mapeamento de Features Brutas para a dimensão de Anomaly Scores
    X_scaled = scaler.transform(df[features_col].values)
    df["Anomaly_Score"] = model.score_samples(X_scaled)
    
    # 2. Suavização temporal por Média Móvel Exponencial (EMA) no score contra ruído
    alpha_ema = 0.2
    df["Anomaly_Score_EMA"] = df["Anomaly_Score"].ewm(alpha=alpha_ema, adjust=False).mean()
    
    # Extração do perfil de referência estável nominal (primeiras 100 janelas estáveis)
    ref_scores = df[df[label_col] == "D0_Baseline"]["Anomaly_Score_EMA"].iloc[:100].values
    
    ks_fps, ks_tps = 0, 0
    psi_fps, psi_tps = 0, 0
    
    # SOLUÇÃO METODOLÓGICA: Avançar com passo iterativo igual a WINDOW_SIZE em vez de passo unitário.
    # Isto remove a sobreposição de janelas de dados e mitiga drasticamente a autocorrelação induzida pelo EMA.
    for i in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):
        janela_atual = df.iloc[i : i + WINDOW_SIZE]
        cenario_ultimo = janela_atual.iloc[-1][label_col]
        scores_correntes = janela_atual["Anomaly_Score_EMA"].values
        
        # Teste A: Kolmogorov-Smirnov no Anomaly Score (Sub-amostrado/Não-sobreposto)
        _, p_value = ks_2samp(ref_scores, scores_correntes)
        if p_value < KS_P_VALUE_TH:
            if cenario_ultimo == "D0_Baseline": ks_fps += 1
            else: ks_tps += 1
            
        # Teste B: PSI no Anomaly Score (Sub-amostrado/Não-sobreposto)
        psi_val = calcular_psi(ref_scores, scores_correntes)
        if psi_val > PSI_THRESHOLD:
            if cenario_ultimo == "D0_Baseline": psi_fps += 1
            else: psi_tps += 1
            
    print(f"DET2 (KS-Test no Score Amostrado) -> Falsos Alarmes: {ks_fps:<4} | Disparos de Drift Verdadeiros: {ks_tps}")
    print(f"DET2 (PSI-Index no Score Amostrado)-> Falsos Alarmes: {psi_fps:<4} | Disparos de Drift Verdadeiros: {psi_tps}")

if __name__ == "__main__":
    print("=== [04] Avaliação Estatística baseada em Anomaly Scores (Correção i.i.d.) ===")
    run_score_based_evaluation("ims_bearing1_features.csv", "Scenario", "Dataset IMS Real Puro")
    run_score_based_evaluation("simulacao_cenario1_regime.csv", "Scenario_Simulado", "Cenário 1: Benign Regime Shift")
    run_score_based_evaluation("simulacao_cenario2_falha.csv", "Scenario_Simulado", "Cenário 2: Severe Fault Degradation")