"""
Script Definitivo: Profiling de Hardware na Edge (Raspberry Pi 5)
Avalia Latência, Retreino Incremental (A2), CPU e RAM com Rigor Estatístico.
Responde aos Pontos 2, 3 e 5 dos Revisores do artigo DriftSense-PM.
"""

import pandas as pd
import numpy as np
import time
import glob
import os
import psutil
import yaml
import subprocess
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- CARREGAR CONFIGURAÇÃO DINAMICAMENTE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/CWRU_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

FEATURES = config['feature_engineering']['features']
PERCENTAGEM_REPLAY = config['adaptation']['percentage_replay']
N_EXECUCOES = config['experiment']['n_executions']
PROCESSED_DIR = get_abs_path(config['paths']['processed_dir'])
RESULTS_DIR = get_abs_path(config['paths']['results_dir'])

# Hiperparâmetros do ficheiro de configuração
OC_SVM_NU = config['models']['oc_svm']['nu']
OC_SVM_GAMMA = config['models']['oc_svm']['gamma']
OC_SVM_KERNEL = config['models']['oc_svm']['kernel']

IF_CONTAMINATION = config['models']['isolation_forest']['contamination']
IF_N_ESTIMATORS = config['models']['isolation_forest']['n_estimators']

LOF_CONTAMINATION = config['models']['local_outlier_factor']['contamination']
LOF_N_NEIGHBORS = config['models']['local_outlier_factor']['n_neighbors']
LOF_NOVELTY = config['models']['local_outlier_factor']['novelty']

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_git_commit():
    """Recupera a hash estável do commit atual do Git."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown_commit"

def carregar_dados_reais():
    print(f"=== A carregar ficheiros reais da pasta '{PROCESSED_DIR}' ===")
    
    ficheiros_csv = glob.glob(os.path.join(PROCESSED_DIR, '*_features.csv'))
    if not ficheiros_csv:
        raise ValueError(f"ERRO: Nenhum ficheiro CSV encontrado em '{PROCESSED_DIR}'.")

    ficheiro_normal = [f for f in ficheiros_csv if 'Normal' in f]
    df_normal = pd.read_csv(ficheiro_normal[0])
    limite = int(len(df_normal) * 0.8)

    X_treino_bruto = df_normal[FEATURES].iloc[:limite].values
    X_teste_normal_bruto = df_normal[FEATURES].iloc[limite:].values

    ficheiros_anomalia = [f for f in ficheiros_csv if 'Normal' not in f]
    dfs_anomalia = [pd.read_csv(f) for f in sorted(ficheiros_anomalia)]
    df_anomalias_juntas = pd.concat(dfs_anomalia, ignore_index=True)
    X_teste_anomalia_bruto = df_anomalias_juntas[FEATURES].values

    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    X_teste_normal = scaler.transform(X_teste_normal_bruto)
    X_teste_anomalia = scaler.transform(X_teste_anomalia_bruto)

    X_teste = np.vstack((X_teste_normal, X_teste_anomalia))

    print(f"-> {len(X_treino)} Janelas de Treino | {len(X_teste)} Janelas de Teste prontas.\n")
    return X_treino, X_teste, X_teste_anomalia

def executar_profiling_edge(X_treino, X_teste, X_teste_anomalia):
    modelos = {
        'One-Class SVM': OneClassSVM(kernel=OC_SVM_KERNEL, nu=OC_SVM_NU, gamma=OC_SVM_GAMMA),
        'Isolation Forest': IsolationForest(contamination=IF_CONTAMINATION, n_estimators=IF_N_ESTIMATORS, random_state=42),
        'Local Outlier Factor': LocalOutlierFactor(contamination=LOF_CONTAMINATION, n_neighbors=LOF_N_NEIGHBORS, novelty=LOF_NOVELTY)
    }

    tamanho_replay = int(len(X_treino) * PERCENTAGEM_REPLAY)
    
    # Random Seed fixada para garantir reprodutibilidade
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    indices_replay = np.random.choice(len(X_treino), size=tamanho_replay, replace=False)
    X_hibrido = np.vstack((X_treino[indices_replay], X_teste_anomalia[:200]))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()
    logs_brutos = []

    print(f"A executar profiling estatístico com {N_EXECUCOES} iterações...\n")
    print(f"{'Modelo':<20} | {'Lat. Inf. (Média ± IC95%)':<27} | {'Retreino (Média ± IC95%)':<27}")
    print("-" * 80)

    for nome, modelo in modelos.items():
        latencias = []
        retreinos = []

        for iteracao in range(N_EXECUCOES):
            modelo.fit(X_treino)

            # Inferência
            t0_inf = time.perf_counter()
            modelo.predict(X_teste)
            t1_inf = time.perf_counter()
            lat_ms = ((t1_inf - t0_inf) / len(X_teste)) * 1000
            latencias.append(lat_ms)

            # Retreino
            if nome == 'Local Outlier Factor':
                modelo_retreino = LocalOutlierFactor(contamination=LOF_CONTAMINATION, n_neighbors=LOF_N_NEIGHBORS, novelty=LOF_NOVELTY)
            else:
                modelo_retreino = modelo

            # Monitorização de Hardware
            mem_antes = get_memory_usage()
            psutil.cpu_percent(interval=None) 

            t0_ret = time.perf_counter()
            modelo_retreino.fit(X_hibrido)
            t1_ret = time.perf_counter()

            cpu_pico = psutil.cpu_percent(interval=None)
            mem_depois = get_memory_usage()
            
            ret_ms = (t1_ret - t0_ret) * 1000
            ram_consumida = max(0.0, mem_depois - mem_antes)
            
            retreinos.append(ret_ms)

            # Guardar registo da iteração atual
            logs_brutos.append({
                'run_id': run_id,
                'git_commit': git_commit,
                'random_seed': RANDOM_SEED,
                'model': nome,
                'iteration': iteracao + 1,
                'latency_ms': lat_ms,
                'retrain_ms': ret_ms,
                'cpu_percent': cpu_pico,
                'ram_mb': ram_consumida
            })

        # Cálculos Estatísticos (IC 95%)
        lat_mean, lat_std = np.mean(latencias), np.std(latencias, ddof=1)
        lat_ci = 1.96 * (lat_std / np.sqrt(N_EXECUCOES))

        ret_mean, ret_std = np.mean(retreinos), np.std(retreinos, ddof=1)
        ret_ci = 1.96 * (ret_std / np.sqrt(N_EXECUCOES))

        str_lat = f"{lat_mean:.4f} ± {lat_ci:.4f} ms"
        str_ret = f"{ret_mean:.2f} ± {ret_ci:.2f} ms"

        print(f"{nome:<20} | {str_lat:<27} | {str_ret:<27}")

    print("-" * 80)
    
    # Guardar os resultados brutos num ficheiro CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    caminho_csv = os.path.join(RESULTS_DIR, 'benchmark_raw.csv')
    df_raw = pd.DataFrame(logs_brutos)
    df_raw.to_csv(caminho_csv, index=False)
    
    print(f"\n[!] Log bruto guardado com sucesso em: {caminho_csv}")

if __name__ == "__main__":
    try:
        X_treino, X_teste, X_teste_anomalia = carregar_dados_reais()
        executar_profiling_edge(X_treino, X_teste, X_teste_anomalia)
    except Exception as e:
        print(f"Erro Crítico: {str(e)}")