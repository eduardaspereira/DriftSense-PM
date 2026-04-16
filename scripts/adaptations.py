import time
import lightgbm as lgb
from sklearn.metrics import f1_score
import pandas as pd

def apply_a0_no_adaptation(model):
    """
    Estratégia A0: None (Zero Adaptação).
    Não faz qualquer alteração ao modelo. 
    Serve como baseline para medir a degradação bruta [cite: 137-139].
    Custo Energético: NULO | Latência: 0 ms
    """
    # print("🛑 A executar A0 (Zero Adaptação) - A registar degradação bruta...")
    latency_ms = 0.0
    return model, latency_ms

def apply_a1_periodic_retrain(model, X_buffer, y_buffer):
    """
    Estratégia A1: Força Bruta (Periodic Retraining). 
    Ignora tudo e faz o treino do zero usando apenas os dados recentes.
    Custo Energético: ALTO | Latência: ALTA
    """
    print(f"🔄 A executar A1 (Retreino Completo) com {len(X_buffer)} amostras...")
    start_time = time.time()
    
    # O scikit-learn wrapper do LightGBM faz reset se chamarmos o .fit() de novo
    model.fit(X_buffer, y_buffer) 
    
    latency_ms = (time.time() - start_time) * 1000
    return model, latency_ms

def apply_a2_lightweight_adapt(old_model, X_buffer, y_buffer):
    """
    Estratégia A2: Fine-Tuning Inteligente (Lightweight Adaptation).
    Pega no modelo antigo e injeta-lhe o conhecimento novo adicionando pequenas árvores.
    Custo Energético: BAIXO | Latência: BAIXA
    """
    print(f"⚡ A executar A2 (Lightweight Adaptation) com {len(X_buffer)} amostras...")
    start_time = time.time()
    
    # Extrair o 'motor' interno do LightGBM
    booster = old_model.booster_
    
    # Criar um dataset leve no formato nativo do LightGBM
    train_data = lgb.Dataset(X_buffer, label=y_buffer)
    
    # O 'init_model' é o segredo aqui. Ele não esquece o passado (D0 a D3), 
    # apenas aprende a lidar com o cenário novo (ex: D4).
    # Vamos adicionar apenas 10 árvores (super rápido).
    new_booster = lgb.train(
        {'objective': 'multiclass', 'num_class': 4, 'verbose': -1}, 
        train_data, 
        num_boost_round=10, 
        init_model=booster
    )
    
    # Devolver o cérebro atualizado para dentro da "carcaça" do Scikit-Learn
    old_model._Booster = new_booster
    
    latency_ms = (time.time() - start_time) * 1000
    return old_model, latency_ms