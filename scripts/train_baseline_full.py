import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# Silenciar avisos desnecessários para um output limpo
warnings.filterwarnings('ignore')

# 1. Configurações de Pastas e Ficheiros
PROCESSED_DIR = '../data/processed/'
FIGURES_DIR = '../results/figures/'
METRICS_DIR = '../results/metrics/'
MODELS_DIR = '../models/'

for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Mapeamento oficial dos teus cenários [cite: 15, 89]
label_map = {
    'D0': 'NORMAL',
    'D1': 'TEMP_FAULT',
    'D2': 'RPM_FAULT',
    'D3': 'NOISE_FAULT'
}

# 2. Carregamento de Dados
print("📂 A carregar janelas de features para o Treino Baseline...")
data_list = []
for file in os.listdir(PROCESSED_DIR):
    prefix = file.split('_')[0]
    if prefix in label_map:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, file))
        df['label'] = label_map[prefix]
        data_list.append(df)

df_final = pd.concat(data_list, ignore_index=True)
X = df_final.drop(['Scenario', 'label'], axis=1)
y_strings = df_final['label']

# --- SOLUÇÃO PARA O ERRO DO XGBOOST ---
# Codifica 'NORMAL', 'TEMP_FAULT', etc. em 0, 1, 2, 3
le = LabelEncoder()
y = le.fit_transform(y_strings)
target_names = le.classes_ # Guarda os nomes originais para os relatórios

# 3. Definição dos Modelos [cite: 583, 592]
lgbm = LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": lgbm,
    "XGBoost": xgb,
    "Ensemble (LGBM+XGB)": VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgb)], voting='soft'
    )
}

# 4. K-Fold Cross Validation (K=5) [cite: 480]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"🔬 A iniciar avaliação comparativa (Total: {len(df_final)} janelas)...")

for name, model in models.items():
    print(f"\n{'='*50}\n# MODELO: {name}\n{'='*40}")
    
    # Previsões robustas via K-Fold
    y_pred = cross_val_predict(model, X, y, cv=kf)
    
    # A) CLASSIFICATION REPORT (Usando target_names para manter o profissionalismo)
    report_str = classification_report(y, y_pred, target_names=target_names, digits=3)
    print("📋 Classification Report:")
    print(report_str)
    
    # Guardar Report
    report_path = os.path.join(METRICS_DIR, f"report_{name.replace(' ', '_').lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"Modelo: {name}\n{report_str}")
    
    # B) MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão: {name}')
    plt.ylabel('Classe Real')
    plt.xlabel('Previsão do Modelo')
    
    fig_path = os.path.join(FIGURES_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fig_path)
    plt.close()
    
    # C) GUARDAR O MODELO E O ENCODER (Apenas LightGBM para o Edge)
    if name == "LightGBM":
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODELS_DIR, 'baseline_model.pkl'))
        joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        print(f"💾 Modelo e Encoder exportados com sucesso.")

print("\n🚀 PIPELINE CONCLUÍDO! Verifica as pastas 'results' e 'models'.")