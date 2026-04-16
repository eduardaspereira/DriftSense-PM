import time
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def apply_a0_no_adaptation(model, scaler):
    """
    Estratégia A0: None (Zero Adaptação).
    Não faz qualquer alteração. Serve como baseline.
    Custo Energético: NULO | Latência: 0 ms
    """
    return model, scaler, 0.0

def apply_a1_periodic_retrain(X_buffer_new, processed_dir):
    """
    Estratégia A1: Força Bruta (Full Retrain).
    Junta o D0 histórico com o buffer novo e treina 100 árvores.
    Custo Energético: ALTO | Latência: ALTA
    """
    start_time = time.time()
    
    # 1. Carregar a Memória Histórica (D0)
    caminho_ref = [f for f in os.listdir(processed_dir) if f.startswith('D0_')][0]
    df_d0 = pd.read_csv(os.path.join(processed_dir, caminho_ref))
    X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
    
    # 2. Preparar os dados novos COM OS NOMES DAS COLUNAS CORRETOS
    df_novos = pd.DataFrame(X_buffer_new, columns=X_d0.columns)
    
    # 3. Juntar Histórico com Novos Dados
    X_combined = pd.concat([X_d0, df_novos], ignore_index=True)
    
    # 4. Novo Scaler e Novo Modelo (Pesado)
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_combined)
    
    new_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    new_model.fit(X_scaled)
    
    latency_ms = (time.time() - start_time) * 1000
    # print(f"🔄 A1 (Retreino Completo) concluído em {latency_ms:.1f}ms. Total amostras: {len(X_combined)}")
    
    return new_model, new_scaler, latency_ms

def apply_a2_lightweight_adapt(X_buffer_new):
    """
    Estratégia A2: Lightweight Adaptation.
    Treina um modelo pequeno APENAS com a nova realidade (buffer).
    Custo Energético: BAIXO | Latência: BAIXA
    """
    start_time = time.time()
    
    # Transformar em DataFrame para o Scaler engolir sem erros de formato
    X_recent = pd.DataFrame(X_buffer_new)
    X_recent.columns = X_recent.columns.astype(str) # Forçar nomes a string
    
    # 1. Scaler adaptado apenas à nova realidade
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_recent)
    
    # 2. Modelo Rápido (Poucas árvores, ideal para a Edge)
    # contamination alta porque assumimos que este pequeno buffer é todo o "novo normal"
    new_model = IsolationForest(n_estimators=10, contamination=0.01, random_state=42)
    new_model.fit(X_scaled)
    
    latency_ms = (time.time() - start_time) * 1000
    # print(f"⚡ A2 (Lightweight) concluído em {latency_ms:.1f}ms.")
    
    return new_model, new_scaler, latency_ms