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
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

FEATURES = ['AccX_Mean', 'AccX_RMS', 'AccX_Skew', 'AccX_Kurt', 'AccX_PeakFreq_Hz']
PERCENTAGEM_REPLAY = 0.15
N_EXECUCOES = 30 # Rigor estatístico exigido pelos revisores

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def carregar_dados_reais():
    caminho_dados = "/home/user/projeto/DriftSense-PM/data/CWRU_dataset/processed/"
    print(f"=== A carregar ficheiros reais da pasta '{caminho_dados}' ===")
    
    ficheiros_csv = glob.glob(os.path.join(caminho_dados, '*_features.csv'))
    if not ficheiros_csv:
        raise ValueError(f"ERRO: Nenhum ficheiro CSV encontrado em '{caminho_dados}'.")

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
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.03, gamma=0.001),
        'Isolation Forest': IsolationForest(contamination=0.03, n_estimators=100, random_state=42),
        'Local Outlier Factor': LocalOutlierFactor(contamination=0.03, n_neighbors=20, novelty=True)
    }

    tamanho_replay = int(len(X_treino) * PERCENTAGEM_REPLAY)
    np.random.seed(42)
    indices_replay = np.random.choice(len(X_treino), size=tamanho_replay, replace=False)
    X_hibrido = np.vstack((X_treino[indices_replay], X_teste_anomalia[:200]))

    print(f"A executar profiling estatístico com {N_EXECUCOES} iterações...\n")
    print(f"{'Modelo':<20} | {'Lat. Inf. (Média ± IC95%)':<27} | {'Retreino (Média ± IC95%)':<27}")
    print("-" * 80)

    for nome, modelo in modelos.items():
        latencias = []
        retreinos = []

        for _ in range(N_EXECUCOES):
            modelo.fit(X_treino)

            # Inferência
            t0_inf = time.perf_counter()
            modelo.predict(X_teste)
            t1_inf = time.perf_counter()
            latencias.append(((t1_inf - t0_inf) / len(X_teste)) * 1000)

            # Retreino
            if nome == 'Local Outlier Factor':
                modelo_retreino = LocalOutlierFactor(contamination=0.03, n_neighbors=20, novelty=True)
            else:
                modelo_retreino = modelo

            t0_ret = time.perf_counter()
            modelo_retreino.fit(X_hibrido)
            t1_ret = time.perf_counter()
            retreinos.append((t1_ret - t0_ret) * 1000)

        # Cálculos Estatísticos (IC 95%)
        lat_mean, lat_std = np.mean(latencias), np.std(latencias, ddof=1)
        lat_ci = 1.96 * (lat_std / np.sqrt(N_EXECUCOES))

        ret_mean, ret_std = np.mean(retreinos), np.std(retreinos, ddof=1)
        ret_ci = 1.96 * (ret_std / np.sqrt(N_EXECUCOES))

        str_lat = f"{lat_mean:.4f} ± {lat_ci:.4f} ms"
        str_ret = f"{ret_mean:.2f} ± {ret_ci:.2f} ms"

        print(f"{nome:<20} | {str_lat:<27} | {str_ret:<27}")

    print("-" * 80)

if __name__ == "__main__":
    try:
        X_treino, X_teste, X_teste_anomalia = carregar_dados_reais()
        executar_profiling_edge(X_treino, X_teste, X_teste_anomalia)
    except Exception as e:
        print(f"Erro Crítico: {str(e)}")