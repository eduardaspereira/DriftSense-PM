#!/usr/bin/env python3
"""
DriftSense-PM: Injeção de Concept Drift Controlado e Multi-Cenário.
Gera os datasets sintéticos fundamentados na cinemática para validação das 
estratégias de veto: Cenário 1 (Regime Shift) e Cenário 2 (Severe Fault).

FUNDAMENTAÇÃO FÍSICA PARA O ARTIGO:
- Delta_Freq (+15.0 Hz): Equivale a um incremento de +900 RPM na rotação do eixo.
- Delta_RMS (*1.15): Proporcional à lei empírica V ~ N^x (onde x=1.5 a 2) para a 
  energia vibratória global decorrente do aumento de velocidade rotacional.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value): return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DATA_DIR = get_abs_path(config['paths']['processed_dir'])
DRIFT_START_IDX = 400  # Ponto de injeção controlado para repetibilidade

# --- PARÂMETROS BASEADOS NA FÍSICA DO PROCESSO ---
RPM_NOMINAL = 2000.0  # Velocidade original do teste IMS (f_shaft = 33.33 Hz)
DELTA_RPM = 900.0     # Alteração operacional simulada (ex: alteração de processo)
DELTA_FREQ_HZ = DELTA_RPM / 60.0  # 900 / 60 = 15.0 Hz de deslocamento síncrono

# Estimativa física do impacto da velocidade no RMS global (relação quadrática típica de energia vibratória V ~ N^2)
FATOR_RMS_CARGA = ( (RPM_NOMINAL + DELTA_RPM) / RPM_NOMINAL ) ** 1.5 # ~1.15x

def carregar_baseline_puro():
    data_path = os.path.join(PROCESSED_DATA_DIR, "ims_bearing1_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Execute o pipeline inicial primeiro. Ausente: {data_path}")
    df = pd.read_csv(data_path)
    return df[df["Scenario"] == "D0_Baseline"].copy().reset_index(drop=True)

def gerar_cenarios_artigo():
    print("=== [04b] A gerar Cenários de Drift Fundamentados na Física ===")
    df_base = carregar_baseline_puro()
    
    # --- CENÁRIO 1: Benign Regime Shift (Mudança de RPM sem Falha) ---
    # Simula alteração na rotação de trabalho justificada cinematicamente
    df_regime = df_base.copy()
    df_regime.loc[DRIFT_START_IDX:, "AccX_PeakFreq_Hz"] += DELTA_FREQ_HZ
    df_regime.loc[DRIFT_START_IDX:, "AccX_RMS"] *= FATOR_RMS_CARGA
    
    df_regime["Scenario_Simulado"] = "D0_Baseline"
    df_regime.loc[DRIFT_START_IDX:, "Scenario_Simulado"] = "Regime_Shift"
    
    path_c1 = os.path.join(PROCESSED_DATA_DIR, "simulacao_cenario1_regime.csv")
    df_regime.to_csv(path_c1, index=False)
    print(f"-> Cenário 1 (Benign Shift de +{DELTA_RPM} RPM) guardado em: {path_c1}")

    # --- CENÁRIO 2: Severe Fault Degradation (Falha Concomitante com Degradação Física) ---
    # Simula o aparecimento de falha severa que impacta a banda de alta frequência por atrito cinemático
    df_falha = df_base.copy()
    n_drift = len(df_falha) - DRIFT_START_IDX
    
    # Ruído incremental simulando a perda de ciclostacionaridade do sinal de vibração
    rampa_ruido = np.linspace(0.0, 0.4, n_drift)
    ruido = np.random.normal(0, rampa_ruido, n_drift)
    
    df_falha.loc[DRIFT_START_IDX:, "Energy_HighFreq"] += ruido
    df_falha.loc[DRIFT_START_IDX:, "AccX_RMS"] += np.linspace(0.0, 0.8, n_drift)
    
    df_falha["Scenario_Simulado"] = "D0_Baseline"
    df_falha.loc[DRIFT_START_IDX:, "Scenario_Simulado"] = "D5_Catastrofico"
    
    path_c2 = os.path.join(PROCESSED_DATA_DIR, "simulacao_cenario2_falha.csv")
    df_falha.to_csv(path_c2, index=False)
    print(f"-> Cenário 2 (Severe Fault Degradation) guardado em: {path_c2}")

if __name__ == "__main__":
    gerar_cenarios_artigo()