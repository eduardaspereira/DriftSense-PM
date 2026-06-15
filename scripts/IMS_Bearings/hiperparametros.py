#!/usr/bin/env python3
"""
Script de Otimização de Hiperparâmetros para o One-Class SVM (DriftSense-PM)
CORREÇÃO DE DATA LEAKAGE: Normalização ajustada estritamente com dados do passado.
INCLUI: Validação Final (Teste Cego) e formatação de resultados para o artigo.
"""

import pandas as pd
import numpy as np
import os
import yaml
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "IMS_Bearings_config.yaml")

# --- FUNÇÃO DE RESOLUÇÃO DE CAMINHOS ---
def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f: 
    config = yaml.safe_load(f)

FEATURES = config['feature_engineering']['features']
PROCESSED_DIR = get_abs_path(config['paths']['processed_dir'])

def carregar_e_separar_dados():
    ficheiro_csv = os.path.join(PROCESSED_DIR, "ims_bearing1_features.csv")
    if not os.path.exists(ficheiro_csv): 
        raise FileNotFoundError(f"ERRO: Ficheiro ausente em: {ficheiro_csv}")

    df = pd.read_csv(ficheiro_csv)
    df_normal = df[df["Scenario"] == "D0_Baseline"]
    
    # Partição cronológica da fatia saudável
    tam_normal = len(df_normal)
    limite_treino = int(tam_normal * 0.6)
    limite_val = int(tam_normal * 0.8)

    X_treino_bruto = df_normal[FEATURES].iloc[:limite_treino].values
    X_val_normal = df_normal[FEATURES].iloc[limite_treino:limite_val].values
    X_teste_normal = df_normal[FEATURES].iloc[limite_val:].values

    df_anomalia = df[df["Scenario"] != "D0_Baseline"]
    X_anomalias_totais = df_anomalia[FEATURES].values

    limite_anomalia_val = int(len(X_anomalias_totais) * 0.2)
    X_val_anomalia = X_anomalias_totais[:limite_anomalia_val]
    X_teste_anomalia = X_anomalias_totais[limite_anomalia_val:]

    X_val = np.vstack((X_val_normal, X_val_anomalia))
    y_val = np.concatenate((np.ones(len(X_val_normal)), np.full(len(X_val_anomalia), -1)))

    X_teste = np.vstack((X_teste_normal, X_teste_anomalia))
    y_teste = np.concatenate((np.ones(len(X_teste_normal)), np.full(len(X_teste_anomalia), -1)))

    # BARREIRA DE LEAKAGE: O Scaler só conhece e processa os dados de X_treino_bruto
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    X_val_scaled = scaler.transform(X_val)
    X_teste_scaled = scaler.transform(X_teste)

    return X_treino, X_val_scaled, y_val, X_teste_scaled, y_teste

def procurar_melhores_parametros():
    X_treino, X_val, y_val, X_teste, y_teste = carregar_e_separar_dados()

    # Grelha fundida: engloba limites menos agressivos do Rasp e os específicos do Colab
    lista_nu = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    lista_gamma = [0.00001, 0.0001, 0.001, 0.005, 0.01]

    resultados = []
    melhor_f1 = 0
    melhores_params = {}

    print("=== A INICIAR VARRIMENTO DE HIPERPARAMETROS (OC-SVM) ===")
    print(f"{'Nu (nu)':<8} | {'Gamma (gamma)':<13} | {'F1-Score Val':<12} | {'FAR (%) Val'}")
    print("-" * 55)
    
    for nu in lista_nu:
        for gamma in lista_gamma:
            modelo = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            modelo.fit(X_treino)
            preds_val = modelo.predict(X_val)

            f1 = f1_score(y_val, preds_val, average='macro')
            cm = confusion_matrix(y_val, preds_val, labels=[1, -1])
            far = (cm[0][1] / (cm[0][1] + cm[0][0])) * 100 if (cm[0][1] + cm[0][0]) > 0 else 0.0
            
            resultados.append({'nu': nu, 'gamma': gamma, 'f1': f1, 'far': far})

            gamma_str = str(gamma)
            print(f"{nu:<8} | {gamma_str:<13} | {f1:<12.4f} | {far:<7.2f}%")

            # Guardar se for o melhor F1-Score e mantiver a FAR civilizada (abaixo de 5%)
            if f1 > melhor_f1 and far < 5.0:
                melhor_f1 = f1
                melhores_params = {'nu': nu, 'gamma': gamma}

    # Ordenar primeiro por F1-Score e depois por FAR para listar o Pareto Front
    resultados.sort(key=lambda x: (x['f1'], -x['far']), reverse=True)
    
    print("\n--- TOP 5 COMPROMISSOS (PARETO FRONT) ---")
    for i, res in enumerate(resultados[:5]):
        print(f"{i+1}. nu={res['nu']}, gamma={res['gamma']} -> F1-Score: {res['f1']:.4f} | FAR: {res['far']:.2f}%")

    # Fallback de segurança: se nenhum modelo tiver FAR < 5%, forçamos o melhor resultado geral
    if not melhores_params:
        melhores_params = {'nu': resultados[0]['nu'], 'gamma': resultados[0]['gamma']}
        melhor_f1 = resultados[0]['f1']
        print("\n[Aviso] Nenhuma configuração obteve FAR < 5.0%. A usar o melhor F1-Score absoluto.")

    print("\n" + "="*60)
    print(" RESULTADO DA OTIMIZAÇÃO HISTÓRICA")
    print("="*60)
    print(f"Melhores parâmetros encontrados: {melhores_params}")
    print(f"Melhor F1-Score na Validação: {melhor_f1:.4f}")

    # --- TESTE CEGO FINAL (Métricas REAIS e imparciais para o Artigo) ---
    print("\n[A Validar no Teste Cego Final para o Artigo...]")
    modelo_final = OneClassSVM(kernel='rbf', **melhores_params)
    modelo_final.fit(X_treino)
    preds_teste = modelo_final.predict(X_teste)

    f1_final = f1_score(y_teste, preds_teste, average='macro')
    cm_final = confusion_matrix(y_teste, preds_teste, labels=[1, -1])
    far_final = (cm_final[0][1] / (cm_final[0][1] + cm_final[0][0])) * 100 if (cm_final[0][1] + cm_final[0][0]) > 0 else 0.0

    print(f"-> F1-Score Global Definitivo: {f1_final:.4f}")
    print(f"-> Taxa de Falsos Alarmes (FAR) Definitiva: {far_final:.2f}%")
    print("="*60)
    print("Copia estes dois valores e os parâmetros para a tua tabela final do artigo!")

if __name__ == "__main__":
    procurar_melhores_parametros()