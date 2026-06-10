"""
Script Definitivo: Benchmark de Detetores de Concept Drift
Compara DET1 (OC-SVM otimizado), DET2 (KS) e ADWIN.
Responde aos Pontos 1, 3 e 4 dos Revisores do artigo DriftSense-PM.
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from river.drift import ADWIN
import warnings

warnings.filterwarnings('ignore')

CAMINHO_NORMAL = 'csvs/Time_Normal_1_098_features.csv'
CAMINHO_FALHA = 'csvs/IR007_1_110_features.csv'
FEATURES = ['AccX_Mean', 'AccX_RMS', 'AccX_Skew', 'AccX_Kurt', 'AccX_PeakFreq_Hz']

DET1_PERSISTENCIA = 10
DET2_TAMANHO_JANELA = 50
DET2_ALPHA = 0.001

def carregar_stream_dados():
    df_normal = pd.read_csv(CAMINHO_NORMAL)[FEATURES]
    df_falha = pd.read_csv(CAMINHO_FALHA)[FEATURES]
    
    # 30% do normal para aprender o estado nominal
    limite_treino = int(len(df_normal) * 0.3)
    X_treino_bruto = df_normal.iloc[:limite_treino].values
    
    # Restante para simular o fluxo contínuo
    stream_normal_bruto = df_normal.iloc[limite_treino:].values
    stream_falha_bruto = df_falha.values
    ponto_drift = len(stream_normal_bruto)
    
    stream_completo_bruto = np.vstack((stream_normal_bruto, stream_falha_bruto))
    
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    stream_completo = scaler.transform(stream_completo_bruto)
    
    return X_treino, stream_completo, ponto_drift

def main():
    X_treino, stream_completo, ponto_drift = carregar_stream_dados()
    
    print(f"Ponto exato do Concept Drift: Janela {ponto_drift}")
    print(f"Total de janelas avaliadas no fluxo: {len(stream_completo)}\n")

    # Inicializar o OC-SVM com a parametrização ótima
    ocsvm = OneClassSVM(kernel='rbf', nu=0.03, gamma=0.001)
    ocsvm.fit(X_treino)
    
    baseline_rms = X_treino[:, 1] # Índice 1 é o AccX_RMS
    janela_deslizante_ks = []
    adwin = ADWIN()

    metricas = {
        'DET1 (OC-SVM)': {'fp': 0, 'delay': None, 'acionou_falha': False, 'erros_consecutivos': 0},
        'DET2 (KS Test)': {'fp': 0, 'delay': None, 'acionou_falha': False},
        'ADWIN (River)': {'fp': 0, 'delay': None, 'acionou_falha': False}
    }

    for i, janela in enumerate(stream_completo):
        is_fase_normal = i < ponto_drift
        
        # 1. DET1 (OC-SVM)
        if not metricas['DET1 (OC-SVM)']['acionou_falha']:
            pred = ocsvm.predict([janela])[0]
            if pred == -1:
                metricas['DET1 (OC-SVM)']['erros_consecutivos'] += 1
            else:
                metricas['DET1 (OC-SVM)']['erros_consecutivos'] = 0
                
            if metricas['DET1 (OC-SVM)']['erros_consecutivos'] >= DET1_PERSISTENCIA:
                if is_fase_normal:
                    metricas['DET1 (OC-SVM)']['fp'] += 1
                    metricas['DET1 (OC-SVM)']['erros_consecutivos'] = 0
                else:
                    metricas['DET1 (OC-SVM)']['delay'] = i - ponto_drift
                    metricas['DET1 (OC-SVM)']['acionou_falha'] = True

        # 2. DET2 (Kolmogorov-Smirnov)
        if not metricas['DET2 (KS Test)']['acionou_falha']:
            janela_deslizante_ks.append(janela[1])
            if len(janela_deslizante_ks) > DET2_TAMANHO_JANELA:
                janela_deslizante_ks.pop(0)
                
            if len(janela_deslizante_ks) == DET2_TAMANHO_JANELA:
                stat, p_value = ks_2samp(baseline_rms, janela_deslizante_ks)
                if p_value < DET2_ALPHA:
                    if is_fase_normal:
                        metricas['DET2 (KS Test)']['fp'] += 1
                        janela_deslizante_ks = []
                    else:
                        metricas['DET2 (KS Test)']['delay'] = i - ponto_drift
                        metricas['DET2 (KS Test)']['acionou_falha'] = True

        # 3. ADWIN (River)
        if not metricas['ADWIN (River)']['acionou_falha']:
            adwin.update(janela[1])
            if adwin.drift_detected:
                if is_fase_normal:
                    metricas['ADWIN (River)']['fp'] += 1
                else:
                    metricas['ADWIN (River)']['delay'] = i - ponto_drift
                    metricas['ADWIN (River)']['acionou_falha'] = True

    print("="*65)
    print(f"{'Mecanismo Detetor':<20} | {'Falsos Alarmes':<15} | {'FAR (%)':<10} | {'Atraso (Janelas)':<10}")
    print("-" * 65)
    
    for nome, dados in metricas.items():
        far = (dados['fp'] / ponto_drift) * 100
        delay = dados['delay'] if dados['delay'] is not None else "Falhou"
        print(f"{nome:<20} | {dados['fp']:<15} | {far:<10.2f} | {delay:<10}")
    print("="*65)

if __name__ == "__main__":
    main()