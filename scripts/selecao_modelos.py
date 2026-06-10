"""
Script Definitivo: Seleção de Modelos Não Supervisionados na Edge (DriftSense-PM)
Treina apenas no regime nominal e testa contra anomalias.
Avalia: F1-Score, FAR (%), Latência de Inferência, Tempo de Retreino (A2), CPU e RAM.
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
from sklearn.metrics import f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# O teu "Quinteto de Ouro" (Maldição da Dimensionalidade resolvida)
FEATURES = ['AccX_Mean', 'AccX_RMS', 'AccX_Skew', 'AccX_Kurt', 'AccX_PeakFreq_Hz']
PERCENTAGEM_REPLAY = 0.15

def get_memory_usage():
    """Retorna o uso atual de RAM do processo em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def carregar_dados_nao_supervisionados():
    print("=== A carregar e a dividir datasets (Treino = Apenas Normal) ===")
    ficheiros_csv = glob.glob('csvs/*_features.csv')

    if not ficheiros_csv:
        raise ValueError("ERRO: Nenhum ficheiro CSV encontrado na pasta 'csvs/'.")

    # 1. Isolar o ficheiro de comportamento Normal
    ficheiro_normal = [f for f in ficheiros_csv if 'Normal' in f]
    if not ficheiro_normal:
        raise ValueError("ERRO: Ficheiro Normal não encontrado.")
        
    df_normal = pd.read_csv(ficheiro_normal[0])
    
    # 2. Dividir o Normal: 80% para ensinar o modelo, 20% para testar Falsos Alarmes
    limite = int(len(df_normal) * 0.8)
    X_treino_bruto = df_normal[FEATURES].iloc[:limite].values
    
    X_teste_normal_bruto = df_normal[FEATURES].iloc[limite:].values
    y_teste_normal = np.ones(len(X_teste_normal_bruto)) # 1 = Saudável

    # 3. Carregar todas as falhas apenas para a fase de TESTE
    ficheiros_anomalia = [f for f in ficheiros_csv if 'Normal' not in f]
    dfs_anomalia = [pd.read_csv(f) for f in sorted(ficheiros_anomalia)]
    df_anomalias_juntas = pd.concat(dfs_anomalia, ignore_index=True)
    
    X_teste_anomalia_bruto = df_anomalias_juntas[FEATURES].values
    y_teste_anomalia = np.full(len(X_teste_anomalia_bruto), -1) # -1 = Falha (Anomalia)

    # 4. Normalização (Escala aprendida APENAS no estado saudável)
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    X_teste_normal = scaler.transform(X_teste_normal_bruto)
    X_teste_anomalia = scaler.transform(X_teste_anomalia_bruto)

    # Construir o vetor de teste final (Mistura do normal que sobrou com as falhas todas)
    X_teste = np.vstack((X_teste_normal, X_teste_anomalia))
    y_teste = np.concatenate((y_teste_normal, y_teste_anomalia))

    print(f"-> Treino: {len(X_treino)} janelas saudáveis.")
    print(f"-> Teste: {len(X_teste)} janelas ({len(X_teste_normal)} normais, {len(X_teste_anomalia)} anómalas).\n")
    
    return X_treino, X_teste, y_teste, X_teste_anomalia

def executar_benchmark_global(X_treino, X_teste, y_teste, X_teste_anomalia):
    # Modelos otimizados das 3 grandes famílias de deteção não supervisionada
    modelos = {
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.05, gamma=0.01),
        'Isolation Forest': IsolationForest(contamination=0.05, n_estimators=100, random_state=42),
        'Local Outlier Factor': LocalOutlierFactor(contamination=0.05, n_neighbors=20, novelty=True)
    }

    # Preparar dados para simular o Retreino Incremental (A2 Híbrido com Replay Buffer)
    tamanho_replay = int(len(X_treino) * PERCENTAGEM_REPLAY)
    np.random.seed(42)
    indices_replay = np.random.choice(len(X_treino), size=tamanho_replay, replace=False)
    X_retreino = np.vstack((X_treino[indices_replay], X_teste_anomalia[:200]))

    print(f"{'Modelo':<20} | {'F1-Score':<8} | {'FAR (%)':<7} | {'Lat. Inf.':<12} | {'Retreino(A2)':<12} | {'CPU %':<6} | {'RAM MB'}")
    print("-" * 105)

    for nome, modelo in modelos.items():
        # --- 1. Treino Inicial (Apenas no Normal) ---
        modelo.fit(X_treino)

        # --- 2. Teste e Inferência (Precisão e FAR) ---
        t0_inf = time.perf_counter()
        y_pred = modelo.predict(X_teste)
        t1_inf = time.perf_counter()
        
        latencia_ms = ((t1_inf - t0_inf) / len(X_teste)) * 1000
        
        # O F1-Score Macro mede o equilíbrio entre acertar no normal e na falha
        f1 = f1_score(y_teste, y_pred, average='macro')
        
        # Calcular Taxa de Falsos Alarmes (Falsos Positivos)
        cm = confusion_matrix(y_teste, y_pred, labels=[1, -1])
        fp = cm[0][1]
        tn = cm[0][0]
        far = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0

        # --- 3. Profiling de Hardware na Adaptação (Estratégia A2) ---
        if nome == 'Local Outlier Factor':
            modelo_retreino = LocalOutlierFactor(contamination=0.05, n_neighbors=20, novelty=True)
        else:
            modelo_retreino = modelo

        mem_antes = get_memory_usage()
        psutil.cpu_percent(interval=None) # Reset do monitor de CPU
        
        t0_ret = time.perf_counter()
        modelo_retreino.fit(X_retreino)
        t1_ret = time.perf_counter()
        
        cpu_pico = psutil.cpu_percent(interval=None)
        mem_depois = get_memory_usage()
        
        tempo_retreino_ms = (t1_ret - t0_ret) * 1000
        ram_consumida = max(0.0, mem_depois - mem_antes)

        print(f"{nome:<20} | {f1:<8.4f} | {far:<7.2f} | {latencia_ms:>7.4f} ms | {tempo_retreino_ms:>7.2f} ms | {cpu_pico:>5.1f} | {ram_consumida:>6.2f}")

    print("-" * 105)

def main():
    try:
        X_treino, X_teste, y_teste, X_teste_anomalia = carregar_dados_nao_supervisionados()
        executar_benchmark_global(X_treino, X_teste, y_teste, X_teste_anomalia)
    except Exception as e:
        print(f"Erro Crítico: {str(e)}")

if __name__ == "__main__":
    main()