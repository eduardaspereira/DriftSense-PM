"""
Descrição: Implementação das estratégias de adaptação (A0, A1, A2).
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Fornece funções que aplicam diferentes políticas de adaptação do modelo:
- A0: nenhuma adaptação (baseline)
- A1: retreino completo com histórico + buffer (custoso)
- A2: adaptação leve apenas com o buffer (rápido)
"""

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
    """Estratégia A0: nenhuma adaptação.

    Retorna o modelo e scaler recebidos sem alterações e latência zero.
    """
    return model, scaler, 0.0

def apply_a1_periodic_retrain(X_buffer_new, processed_dir):
    """Estratégia A1: retreino completo com histórico (custo elevado).

    Junta o histórico (D0) com as amostras novas do buffer, treina um novo
    `IsolationForest` com mais estimadores e devolve o novo par
    (modelo, scaler) além da latência em ms.
    """
    start_time = time.time()

    # 1. Carregar a Memória Histórica (D0)
    caminho_ref = [f for f in os.listdir(processed_dir) if f.startswith('D0_')][0]
    df_d0 = pd.read_csv(os.path.join(processed_dir, caminho_ref))
    X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')

    # 2. Preparar os dados novos com as mesmas colunas
    df_novos = pd.DataFrame(X_buffer_new, columns=X_d0.columns)

    # 3. Juntar Histórico com Novos Dados
    X_combined = pd.concat([X_d0, df_novos], ignore_index=True)

    # 4. Novo Scaler e Novo Modelo (pesado)
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_combined)

    new_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    new_model.fit(X_scaled)

    latency_ms = (time.time() - start_time) * 1000
    return new_model, new_scaler, latency_ms

def apply_a2_lightweight_adapt(X_buffer_new):
    """Estratégia A2: adaptação leve apenas com o buffer (baixo custo).

    Treina um `IsolationForest` pequeno utilizando só as amostras recentes
    (buffer). Ideal para execução em Edge com latência reduzida.
    """
    start_time = time.time()

    # 1. Scaler adaptado apenas à nova realidade
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_buffer_new)

    # 2. Modelo rápido (poucas árvores)
    new_model = IsolationForest(n_estimators=10, contamination=0.01, random_state=42)
    new_model.fit(X_scaled)

    latency_ms = (time.time() - start_time) * 1000
    return new_model, new_scaler, latency_ms