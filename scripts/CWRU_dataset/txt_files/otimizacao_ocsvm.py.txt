"""
Script de Otimização de Hiperparâmetros para o One-Class SVM (DriftSense-PM)
Executa um varrimento estatístico rigoroso (Grid Search) respeitando a premissa não supervisionada.
"""

import pandas as pd
import numpy as np
import glob
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

FEATURES = ['AccX_Mean', 'AccX_RMS', 'AccX_Skew', 'AccX_Kurt', 'AccX_PeakFreq_Hz']

def carregar_e_separar_dados():
    ficheiros_csv = glob.glob('data/CWRU_dataset/processed/*_features.csv')
    if not ficheiros_csv:
        raise ValueError("ERRO: Nenhum ficheiro CSV encontrado na pasta 'data/CWRU_dataset/processed/'.")

    ficheiro_normal = [f for f in ficheiros_csv if 'Normal' in f]
    df_normal = pd.read_csv(ficheiro_normal[0])
    
    # --- DIVISÃO DO JOGO DE DADOS SAUDÁVEL ---
    # 60% Treino Puro, 20% Validação (Afinamento), 20% Teste Final
    tam_normal = len(df_normal)
    limite_treino = int(tam_normal * 0.6)
    limite_val = int(tam_normal * 0.8)
    
    X_treino_bruto = df_normal[FEATURES].iloc[:limite_treino].values
    
    X_val_normal = df_normal[FEATURES].iloc[limite_treino:limite_val].values
    X_teste_normal = df_normal[FEATURES].iloc[limite_val:].values

    # --- CARREGAR AS ANOMALIAS ---
    ficheiros_anomalia = [f for f in ficheiros_csv if 'Normal' not in f]
    dfs_anomalia = [pd.read_csv(f) for f in sorted(ficheiros_anomalia)]
    df_anomalias_juntas = pd.concat(dfs_anomalia, ignore_index=True)
    X_anomalias_totais = df_anomalias_juntas[FEATURES].values
    
    # Dividir as anomalias: 20% para validação e 80% para o teste definitivo do artigo
    limite_anomalia_val = int(len(X_anomalias_totais) * 0.2)
    X_val_anomalia = X_anomalias_totais[:limite_anomalia_val]
    X_teste_anomalia = X_anomalias_totais[limite_anomalia_val:]

    # --- CONSTRUIR MATRIZES DE VALIDAÇÃO E TESTE ---
    X_val = np.vstack((X_val_normal, X_val_anomalia))
    y_val = np.concatenate((np.ones(len(X_val_normal)), np.full(len(X_val_anomalia), -1)))
    
    X_teste = np.vstack((X_teste_normal, X_teste_anomalia))
    y_teste = np.concatenate((np.ones(len(X_teste_normal)), np.full(len(X_teste_anomalia), -1)))

    # --- NORMALIZAÇÃO ---
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino_bruto)
    X_val_scaled = scaler.transform(X_val)
    X_teste_scaled = scaler.transform(X_teste)

    return X_treino, X_val_scaled, y_val, X_teste_scaled, y_teste

def procurar_melhores_parametros():
    X_treino, X_val, y_val, X_teste, y_teste = carregar_e_separar_dados()
    
    # Espaço de busca (Grid) recomendado para rolamentos e vibração industrial
    lista_nu = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
    lista_gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 'scale']
    
    melhor_f1 = 0
    melhores_params = {}
    
    print("=== A INICIAR VARRIMENTO DE HIPERPARAMETROS (OC-SVM) ===")
    print(f"{'Nu (nu)':<8} | {'Gamma (gamma)':<13} | {'F1-Score Val':<12} | {'FAR (%) Val'}")
    print("-" * 55)
    
    for nu in lista_nu:
        for gamma in lista_gamma:
            # Inicializar o modelo com a combinação atual
            modelo = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            
            # Treinar APENAS com dados saudáveis
            modelo.fit(X_treino)
            
            # Prever no conjunto de validação (onde avaliamos o ajuste)
            preds_val = modelo.predict(X_val)
            
            # Calcular métricas de validação
            f1 = f1_score(y_val, preds_val, average='macro')
            cm = confusion_matrix(y_val, preds_val, labels=[1, -1])
            far = (cm[0][1] / (cm[0][1] + cm[0][0])) * 100 if (cm[0][1] + cm[0][0]) > 0 else 0.0
            
            gamma_str = str(gamma)
            print(f"{nu:<8} | {gamma_str:<13} | {f1:<12.4f} | {far:<7.2f}%")
            
            # Guardar se for o melhor F1-Score e mantiver a FAR civilizada (abaixo de 5%)
            if f1 > melhor_f1 and far < 5.0:
                melhor_f1 = f1
                melhores_params = {'nu': nu, 'gamma': gamma}
                
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
    far_final = (cm_final[0][1] / (cm_final[0][1] + cm_final[0][0])) * 100
    
    print(f"-> F1-Score Global Definitivo: {f1_final:.4f}")
    print(f"-> Taxa de Falsos Alarmes (FAR) Definitiva: {far_final:.2f}%")
    print("="*60)
    print("Copia estes dois valores e os parâmetros para a tua tabela final do artigo!")

if __name__ == "__main__":
    procurar_melhores_parametros()