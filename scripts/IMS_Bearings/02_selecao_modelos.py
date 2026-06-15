#!/usr/bin/env python3
"""
Script Definitivo: Seleção de Modelos Não Supervisionados na Edge (IMS Bearings)
Avaliação estatística com rigor científico sobre 30 execuções independentes (IC95%).
CORREÇÃO DE LEAKAGE: O Replay Buffer usa apenas dados do passado histórico nominal.
"""

import pandas as pd
import numpy as np
import time
import json
import os
import psutil
import yaml
from scipy.stats import t
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value): return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)

FEATURES = config['feature_engineering']['features']
PERCENTAGEM_REPLAY = config['adaptation']['percentage_replay']
PROCESSED_DIR = get_abs_path(config['paths']['processed_dir'])
N_EXECUTIONS = config['experiment'].get('n_executions', 30)

OC_SVM_NU = config['models']['oc_svm']['nu']
OC_SVM_GAMMA = config['models']['oc_svm']['gamma']
OC_SVM_KERNEL = config['models']['oc_svm']['kernel']
IF_CONTAMINATION = config['models']['isolation_forest']['contamination']
IF_N_ESTIMATORS = config['models']['isolation_forest']['n_estimators']
LOF_CONTAMINATION = config['models']['local_outlier_factor']['contamination']
LOF_N_NEIGHBORS = config['models']['local_outlier_factor']['n_neighbors']
LOF_NOVELTY = config['models']['local_outlier_factor']['novelty']

def carregar_dados_ims():
    ficheiro_csv = os.path.join(PROCESSED_DIR, "ims_bearing1_features.csv")
    if not os.path.exists(ficheiro_csv):
        raise ValueError(f"ERRO: Ficheiro não encontrado em '{ficheiro_csv}'.")
    
    df = pd.read_csv(ficheiro_csv)
    df_normal = df[df["Scenario"] == "D0_Baseline"]
    
    # Divisão estritamente cronológica para evitar misturar dados futuros na normalização
    limite = int(len(df_normal) * 0.8)
    X_treino_bruto = df_normal[FEATURES].iloc[:limite].values
    X_teste_normal_bruto = df_normal[FEATURES].iloc[limite:].values
    
    y_teste_normal = np.ones(len(X_teste_normal_bruto))
    df_anomalia = df[df["Scenario"] != "D0_Baseline"]
    X_teste_anomalia_bruto = df_anomalia[FEATURES].values
    y_teste_anomalia = np.full(len(X_teste_anomalia_bruto), -1)
    
    # Ajuste do Scaler feito estritamente no passado saudável (X_treino_bruto)
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    X_teste_normal = scaler.transform(X_teste_normal_bruto)
    X_teste_anomalia = scaler.transform(X_teste_anomalia_bruto)
    
    X_teste = np.vstack((X_teste_normal, X_teste_anomalia))
    y_teste = np.concatenate((y_teste_normal, y_teste_anomalia))
    return X_treino, X_teste, y_teste

def calcular_ic95(dados):
    n = len(dados)
    if n < 2: return 0.0
    media_erro_padrao = np.std(dados, ddof=1) / np.sqrt(n)
    return float(t.ppf(0.975, df=n-1) * media_erro_padrao)

def executar_benchmark_global(X_treino, X_teste, y_teste):
    nomes_modelos = ['One-Class SVM', 'Isolation Forest', 'Local Outlier Factor']
    historico_runs = {nome: {"f1": [], "latencia": [], "far": [], "tempo_ret": []} for nome in nomes_modelos}
    tamanho_replay = int(len(X_treino) * PERCENTAGEM_REPLAY)
    
    print(f"=== Benchmarking Estatístico Sem Fuga de Dados ({N_EXECUTIONS} Runs) ===")
    
    for run in range(N_EXECUTIONS):
        run_seed = config['experiment']['random_seed'] + run
        np.random.seed(run_seed)
        
        # Correção Metodológica: Replay Buffer amostra apenas do histórico de treino disponível (passado)
        indices_replay = np.random.choice(len(X_treino), size=tamanho_replay, replace=False)
        X_retreino = X_treino[indices_replay] 
        
        modelos = {
            'One-Class SVM': OneClassSVM(kernel=OC_SVM_KERNEL, nu=OC_SVM_NU, gamma=OC_SVM_GAMMA),
            'Isolation Forest': IsolationForest(contamination=IF_CONTAMINATION, n_estimators=IF_N_ESTIMATORS, random_state=run_seed),
            'Local Outlier Factor': LocalOutlierFactor(contamination=LOF_CONTAMINATION, n_neighbors=LOF_N_NEIGHBORS, novelty=LOF_NOVELTY)
        }
        
        for nome, modelo in modelos.items():
            modelo.fit(X_treino)
            
            t0_inf = time.perf_counter()
            y_pred = modelo.predict(X_teste)
            t1_inf = time.perf_counter()
            
            lat_ms = ((t1_inf - t0_inf) / len(X_teste)) * 1000
            f1 = f1_score(y_teste, y_pred, average='macro')
            cm = confusion_matrix(y_teste, y_pred, labels=[1, -1])
            far = (cm[0][1] / (cm[0][1] + cm[0][0])) * 100 if (cm[0][1] + cm[0][0]) > 0 else 0.0
            
            modelo_retreino = LocalOutlierFactor(contamination=LOF_CONTAMINATION, n_neighbors=LOF_N_NEIGHBORS, novelty=LOF_NOVELTY) if nome == 'Local Outlier Factor' else modelo
            
            t0_ret = time.perf_counter()
            modelo_retreino.fit(X_retreino)
            t1_ret = time.perf_counter()
            ret_ms = (t1_ret - t0_ret) * 1000
            
            historico_runs[nome]["f1"].append(f1)
            historico_runs[nome]["latencia"].append(lat_ms)
            historico_runs[nome]["far"].append(far)
            historico_runs[nome]["tempo_ret"].append(ret_ms)

    dados_pareto = []
    print(f"\n{'Modelo':<20} | {'F1-Score (±IC95%)':<22} | {'Latência (ms)':<16} | {'FAR (%)'}")
    print("-" * 80)
    
    for nome in nomes_modelos:
        m_f1, ic_f1 = np.mean(historico_runs[nome]["f1"]), calcular_ic95(historico_runs[nome]["f1"])
        m_lat, ic_lat = np.mean(historico_runs[nome]["latencia"]), calcular_ic95(historico_runs[nome]["latencia"])
        m_far = np.mean(historico_runs[nome]["far"])
        m_ret, ic_ret = np.mean(historico_runs[nome]["tempo_ret"]), calcular_ic95(historico_runs[nome]["tempo_ret"])
        
        print(f"{nome:<20} | {m_f1:.4f} ± {ic_f1:.4f}  | {m_lat:.4f} ms       | {m_far:.2f}%")
        dados_pareto.append({
            "modelo": nome, "f1_score": round(m_f1, 4), "f1_ic95": round(ic_f1, 4),
            "latencia_inferencia_ms": round(m_lat, 4), "far_percentagem": round(m_far, 2), "tempo_retreino_ms": round(m_ret, 2)
        })
        
    target_results_dir = get_abs_path(config['paths']['results_dir'])
    os.makedirs(target_results_dir, exist_ok=True)
    with open(os.path.join(target_results_dir, "selecao_modelos_pareto.json"), "w") as f:
        json.dump(dados_pareto, f, indent=4)

if __name__ == "__main__":
    X_treino, X_teste, y_teste = carregar_dados_ims()
    executar_benchmark_global(X_treino, X_teste, y_teste)