"""
Script Definitivo: Profiling de Hardware na Edge (Raspberry Pi 5)
Avalia Latência, Retreino Incremental (A2), CPU e RAM.
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

# As 5 features trancadas para evitar a Maldição da Dimensionalidade
FEATURES = ['AccX_Mean', 'AccX_RMS', 'AccX_Skew', 'AccX_Kurt', 'AccX_PeakFreq_Hz']
PERCENTAGEM_REPLAY = 0.15

def get_memory_usage():
    """Retorna o uso atual de memória RAM do processo em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def carregar_dados_reais():
    print("=== A carregar ficheiros reais da pasta 'csvs/' ===")
    ficheiros_csv = glob.glob('csvs/*_features.csv')

    if not ficheiros_csv:
        raise ValueError("ERRO: Nenhum ficheiro CSV encontrado na pasta 'csvs/'.")

    ficheiro_normal = [f for f in ficheiros_csv if 'Normal' in f]
    if not ficheiro_normal:
        raise ValueError("ERRO: Ficheiro Normal não encontrado.")
        
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
    # Modelos calibrados (OC-SVM com os hiperparâmetros provados empiricamente)
    modelos = {
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.03, gamma=0.001),
        'Isolation Forest': IsolationForest(contamination=0.03, n_estimators=100, random_state=42),
        'Local Outlier Factor': LocalOutlierFactor(contamination=0.03, n_neighbors=20, novelty=True)
    }

    # Preparar Replay Buffer para o teste de Retreino Incremental (A2 Híbrido)
    tamanho_replay = int(len(X_treino) * PERCENTAGEM_REPLAY)
    np.random.seed(42)
    indices_replay = np.random.choice(len(X_treino), size=tamanho_replay, replace=False)
    X_hibrido = np.vstack((X_treino[indices_replay], X_teste_anomalia[:200]))

    print(f"{'Modelo':<22} | {'Lat. Inferência':<17} | {'Retreino (A2)':<15} | {'CPU (Pico)':<12} | {'RAM Alocada'}")
    print("-" * 95)

    for nome, modelo in modelos.items():
        # Treino Base Inicial
        modelo.fit(X_treino)

        # Medir Latência de Inferência Rigorosa
        t0_inf = time.perf_counter()
        modelo.predict(X_teste)
        t1_inf = time.perf_counter()
        latencia_ms = ((t1_inf - t0_inf) / len(X_teste)) * 1000

        # Simular e medir o Retreino Incremental na Edge
        if nome == 'Local Outlier Factor':
            modelo_retreino = LocalOutlierFactor(contamination=0.03, n_neighbors=20, novelty=True)
        else:
            modelo_retreino = modelo

        mem_antes = get_memory_usage()
        psutil.cpu_percent(interval=None) 
        
        t0_ret = time.perf_counter()
        modelo_retreino.fit(X_hibrido)
        t1_ret = time.perf_counter()
        
        cpu_pico = psutil.cpu_percent(interval=None)
        mem_depois = get_memory_usage()
        
        tempo_retreino_ms = (t1_ret - t0_ret) * 1000
        ram_consumida = max(0.0, mem_depois - mem_antes)

        print(f"{nome:<22} | {latencia_ms:>14.4f} ms | {tempo_retreino_ms:>12.2f} ms | {cpu_pico:>8.1f} % | {ram_consumida:>8.2f} MB")

    print("-" * 95)

if __name__ == "__main__":
    try:
        X_treino, X_teste, X_teste_anomalia = carregar_dados_reais()
        executar_profiling_edge(X_treino, X_teste, X_teste_anomalia)
    except Exception as e:
        print(f"Erro Crítico: {str(e)}")