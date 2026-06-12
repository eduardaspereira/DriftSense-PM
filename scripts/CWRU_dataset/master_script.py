#!/usr/bin/env python3
"""
DriftSense-PM: Full Factorial Master Script (CWRU Dataset Validation)
ACM-Ready: Automated reproducibility, hardware telemetry, and artifact logging.
"""

import pandas as pd
import numpy as np
import os
import time
import argparse
import psutil
import subprocess
import uuid
import yaml
import warnings
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from river.drift import ADWIN

warnings.filterwarnings('ignore')

# --- 1. SETUP E METADADOS ACM ---
def get_git_commit():
    """Captura a hash do commit atual para garantir reprodutibilidade (ACM)."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown_commit"

GIT_COMMIT = get_git_commit()
DEVICE_ID = "RaspberryPi5_8GB"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Obter o diretório onde o script está localizado para ancorar caminhos dinâmicos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/CWRU_config.yaml")

parser = argparse.ArgumentParser(description='CWRU Full Factorial Edge Evaluation')
parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to config file')
parser.add_argument('--repetitions', type=int, default=30, help='Number of independent runs')
args = parser.parse_args()

# --- 2. CARREGAR CONFIGURAÇÕES ---
try:
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Erro: Ficheiro {args.config} não encontrado.")
    exit()

def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

# Ajuste estrito dos caminhos e parâmetros baseados no ficheiro CWRU_config.yaml
PROCESSED_DIR = get_abs_path(config['paths']['processed_dir'])
RESULTS_DIR = get_abs_path(config['paths']['results_dir'])
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = config['feature_engineering']['features']
NU = config['models']['oc_svm']['nu']
GAMMA = config['models']['oc_svm']['gamma']
REPLAY_PERCENTAGE = config['adaptation']['percentage_replay']


# Mapeamento completo e exaustivo de todos os ficheiros processados na pasta
SCENARIOS = {
    # Baseline de Controlo
    "D0_Baseline": "Time_Normal_1_098_features.csv",
    
    # Falhas na Pista Interna (Inner Race)
    "D1_IR_Ligeira_7mil": "IR007_1_110_features.csv",
    "D1_IR_Media_14mil": "IR014_1_175_features.csv",
    "D1_IR_Severa_21mil": "IR021_1_214_features.csv",
    
    # Falhas nos Elementos Rolantes / Esferas (Ball)
    "D2_B_Ligeira_7mil": "B007_1_123_features.csv",
    "D2_B_Media_14mil": "B014_1_190_features.csv",
    "D2_B_Severa_21mil": "B021_1_227_features.csv",
    
    # Falhas na Pista Externa (Outer Race)
    "D3_OR_Ligeira_7mil": "OR007_6_1_136_features.csv",
    "D3_OR_Media_14mil": "OR014_6_1_202_features.csv",
    "D3_OR_Severa_21mil": "OR021_6_1_239_features.csv"
}

# --- 3. MOTOR DE SIMULAÇÃO (STREAMING E EDGE TELEMETRY) ---
def simulate_stream_cwru(scenario_name, drift_file, detector_type, adaptation_type):
    """Simula o fluxo temporal exato unindo o regime normal à anomalia injetada."""
    
    # 1. Carregar Dados e Criar Stream
    df_normal = pd.read_csv(os.path.join(PROCESSED_DIR, SCENARIOS["D0_Baseline"]))[FEATURES]
    
    limite_treino = int(len(df_normal) * 0.3)
    X_treino_bruto = df_normal.iloc[:limite_treino].values
    stream_normal = df_normal.iloc[limite_treino:].values
    
    if scenario_name == "D0_Baseline":
        stream_drift = np.empty((0, len(FEATURES)))
    else:
        stream_drift = pd.read_csv(os.path.join(PROCESSED_DIR, drift_file))[FEATURES].values

    ponto_drift_real = len(stream_normal)
    stream_completo_bruto = np.vstack((stream_normal, stream_drift))
    
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    stream_completo = scaler.transform(stream_completo_bruto)

    # 2. Inicializar Modelo e Detetores
    model = OneClassSVM(kernel='rbf', nu=NU, gamma=GAMMA)
    model.fit(X_treino)
    
    baseline_rms = X_treino[:, 1]
    janela_ks = []
    adwin = ADWIN()
    
    # Métricas de acompanhamento
    detection_idx = None
    falsos_alarmes = 0
    erros_consecutivos = 0
    
    tempo_adaptacao_ms = 0.0
    cpu_pico = 0.0
    ram_alocada = 0.0
    adapted_once = False

    # 3. Processamento Contínuo (Simulação Edge)
    for i, janela in enumerate(stream_completo):
        is_normal_phase = i < ponto_drift_real
        y_pred = model.predict([janela])[0]
        
        # --- LÓGICA DE DETEÇÃO ---
        drift_sinalizado = False
        
        if detector_type == 'DET1_OCSVM' and detection_idx is None:
            if y_pred == -1: erros_consecutivos += 1
            else: erros_consecutivos = 0
            
            if erros_consecutivos >= 10: # Persistência
                drift_sinalizado = True
                
        elif detector_type == 'DET2_KS' and detection_idx is None:
            janela_ks.append(janela[1])
            if len(janela_ks) > 50: janela_ks.pop(0)
            if len(janela_ks) == 50:
                _, p_val = ks_2samp(baseline_rms, janela_ks)
                if p_val < 0.001: drift_sinalizado = True
                
        elif detector_type == 'DET3_ADWIN' and detection_idx is None:
            adwin.update(janela[1])
            if adwin.drift_detected: drift_sinalizado = True

        # Processar o alarme
        if drift_sinalizado:
            if is_normal_phase:
                falsos_alarmes += 1
                erros_consecutivos = 0
                janela_ks = []
            else:
                detection_idx = i
        
        # --- LÓGICA DE ADAPTAÇÃO ---
        if detection_idx is not None and not adapted_once and adaptation_type != 'A0_None':
            process = psutil.Process(os.getpid())
            mem_antes = process.memory_info().rss / (1024 * 1024)
            psutil.cpu_percent(interval=None)
            
            t0 = time.perf_counter()
            
            if adaptation_type == 'A1_Global':
                # Retreino Global massivo (Treino + Histórico recente)
                X_retreino = np.vstack((X_treino, stream_completo[i-200:i]))
                model.fit(X_retreino)
                
            elif adaptation_type == 'A2_Incremental':
                # Estratégia Híbrida Leve (Replay Buffer + Anomalia)
                tamanho_replay = int(len(X_treino) * REPLAY_PERCENTAGE)
                indices_replay = np.random.choice(len(X_treino), size=tamanho_replay, replace=False)
                X_retreino = np.vstack((X_treino[indices_replay], stream_completo[i-100:i]))
                model.fit(X_retreino)

            t1 = time.perf_counter()
            cpu_pico = psutil.cpu_percent(interval=None)
            mem_depois = process.memory_info().rss / (1024 * 1024)
            
            tempo_adaptacao_ms = (t1 - t0) * 1000
            ram_alocada = max(0.0, mem_depois - mem_antes)
            adapted_once = True

    # Cálculos finais
    atraso_real = (detection_idx - ponto_drift_real) if detection_idx is not None else -1
    far_percentage = (falsos_alarmes / ponto_drift_real) * 100 if ponto_drift_real > 0 else 0.0
    
    return atraso_real, far_percentage, falsos_alarmes, tempo_adaptacao_ms, cpu_pico, ram_alocada

# --- 4. EXECUÇÃO DA MATRIZ FATORIAL ---
def main():
    detectors = ['DET0_None', 'DET1_OCSVM', 'DET2_KS', 'DET3_ADWIN']
    adaptations = ['A0_None', 'A1_Global', 'A2_Incremental']
    results = []
    
    print("\n" + "="*80)
    print(f"DRIFTSENSE-PM: MATRIZ FATORIAL CWRU")
    print(f"Dispositivo: {DEVICE_ID} | Semente: {RANDOM_SEED} | Repetições: {args.repetitions}")
    print("="*80)

    total_runs = len(SCENARIOS) * len(detectors) * len(adaptations) * args.repetitions
    current_run = 0

    for scenario_name, file_name in SCENARIOS.items():
        for det in detectors:
            for adapt in adaptations:
                # Lógicas excludentes (A0 só faz sentido sem adaptação, e DET0 não adapta)
                if det == 'DET0_None' and adapt != 'A0_None': continue
                if adapt != 'A0_None' and det == 'DET0_None': continue
                
                for rep in range(args.repetitions):
                    current_run += 1
                    run_id = str(uuid.uuid4())[:8]
                    
                    print(f"[{current_run}/{total_runs}] A processar: {scenario_name} | {det} | {adapt} | Rep: {rep+1}")
                    
                    if det == 'DET0_None':
                        atraso, far, fps, t_adapt, cpu, ram = -1, 0.0, 0, 0.0, 0.0, 0.0
                    else:
                        atraso, far, fps, t_adapt, cpu, ram = simulate_stream_cwru(scenario_name, file_name, det, adapt)
                        
                    results.append({
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Run_ID': run_id,
                        'Git_Commit': GIT_COMMIT,
                        'Scenario': scenario_name,
                        'Detector': det,
                        'Adaptation': adapt,
                        'Repetition': rep + 1,
                        'Delay_Windows': atraso if atraso != -1 else "N/D",
                        'False_Alarms': fps,
                        'FAR_%': round(far, 4),
                        'Adapt_Latency_ms': round(t_adapt, 4),
                        'CPU_Peak_%': round(cpu, 2),
                        'RAM_Alloc_MB': round(ram, 4)
                    })

    # Exportar resultados de artefacto ACM
    df_results = pd.DataFrame(results)
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = os.path.join(RESULTS_DIR, f'factorial_matrix_{timestamp_file}.csv')
    df_results.to_csv(output_csv, index=False)
    
    print("\n" + "="*80)
    print(f"AVALIAÇÃO CONCLUÍDA. Log exportado para: {output_csv}")
    print("="*80)

if __name__ == "__main__":
    main()