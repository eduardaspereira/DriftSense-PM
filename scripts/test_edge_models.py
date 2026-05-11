import os
import yaml
import pandas as pd
import numpy as np
import joblib
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost não instalado - será omitido do teste")

warnings.filterwarnings('ignore')

# 1. CONFIGURAÇÕES
CONFIG_PATH = "../configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config['paths']['processed_dir']
METRICS_DIR = config['paths']['results_dir']
MODELS_DIR = config['paths']['models_dir']

for folder in [METRICS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

target_names = ['Anomalia/Drift (-1)', 'Normal (1)']

# 2. CARREGAMENTO E SPLIT (IGUAL AO SVM ORIGINAL)
print("📂 Carregando dados com MESMA configuração do SVM original...\n")
caminho_d0 = os.path.join(PROCESSED_DIR, "D0_dataset_features.csv")
df_d0 = pd.read_csv(caminho_d0)

X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
y_d0 = np.ones(len(X_d0))  # 1 = Normal

# Split IGUAL ao SVM: 80% treino (só normais), 20% teste normais
X_train, X_test_normal, y_train, y_test_normal = train_test_split(X_d0, y_d0, test_size=0.2, shuffle=False)

# Carregar dados de drift para teste (IGUAL ao SVM)
test_anomalies = []
for file in os.listdir(PROCESSED_DIR):
    if file.endswith(".csv") and not file.startswith("D0"):
        df_anom = pd.read_csv(os.path.join(PROCESSED_DIR, file))
        X_anom = df_anom.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
        y_anom = np.full(len(X_anom), -1)
        test_anomalies.append((X_anom, y_anom))

X_test = pd.concat([X_test_normal] + [anom[0] for anom in test_anomalies], ignore_index=True)
y_test = np.concatenate([y_test_normal] + [anom[1] for anom in test_anomalies])

# Converter para 0/1 para tree-based models (manter -1/1 para SVM em computação)
y_train_trees = np.where(y_train == -1, 0, 1)
y_test_trees = np.where(y_test == -1, 0, 1)

print(f"✅ Dataset (IGUAL SVM original):")
print(f"   Treino: {len(X_train)} amostras (D0 - só normais)")
print(f"   Teste: {len(X_test)} amostras ({len(X_test_normal)} normais + {len(y_test)-len(X_test_normal)} drifts)\n")

# 3. NORMALIZAÇÃO
print("⚖️ Normalizando features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. DEFINIÇÃO DE MODELOS
models = {
    "One-Class SVM (Baseline)": OneClassSVM(nu=0.01, kernel="rbf", gamma='scale'),
    "Decision Tree (One-Class)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest (Lightweight)": RandomForestClassifier(n_estimators=5, max_depth=8, random_state=42, n_jobs=1),
    "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.01, random_state=42),
}

# Adicionar XGBoost se disponível
if HAS_XGBOOST:
    models["XGBoost (Shallow)"] = xgb.XGBClassifier(
        n_estimators=10, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=10  # Balancear classes
    )

print(f"🔬 Testando {len(models)} modelos em {len(X_test)} amostras de teste...\n")
print("="*100)

results = []

for name, model in models.items():
    print(f"\n# MODELO: {name}")
    print("-" * 100)
    
    # Treino: TODOS como one-class (treinar só com normais)
    start_train = time.time()
    
    if "SVM" in name or "Isolation Forest" in name:
        # One-Class SVM e Isolation Forest: treinar diretamente com X
        model.fit(X_train_scaled)
    else:
        # Decision Tree e Random Forest: treinar como one-class 
        # (usar labels binários: 1=normal, 0=anomalia, mas treinar só com 1)
        model.fit(X_train_scaled, np.ones(len(X_train_scaled)))
    
    train_time = time.time() - start_train
    
    # Inferência (com timing)
    start_infer = time.time()
    y_pred_raw = model.predict(X_test_scaled)
    infer_time = time.time() - start_infer
    infer_time_per_sample = (infer_time / len(X_test)) * 1000  # ms
    
    # Converter para formato -1/1 (anomalia/normal)
    if "Isolation Forest" in name:
        # Isolation Forest: 1=inlier, -1=outlier (já está no formato certo)
        y_pred_svm_format = y_pred_raw
    else:
        # SVM, DT, RF: 1=inlier, -1=outlier
        # ou predict 0/1, converter para -1/1
        if y_pred_raw.min() >= 0:  # 0/1 format
            y_pred_svm_format = np.where(y_pred_raw == 1, 1, -1)
        else:  # -1/1 format
            y_pred_svm_format = y_pred_raw
    
    f1 = f1_score(y_test, y_pred_svm_format, average='weighted')
    report = classification_report(y_test, y_pred_svm_format, target_names=target_names, digits=3)
    
    # Tamanho do modelo (em KB)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
    joblib.dump(model, tmp_path)
    model_size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    
    print(f"⏱️  Tempo de Treino: {train_time:.3f}s")
    print(f"⏱️  Tempo de Inferência: {infer_time:.3f}s ({infer_time_per_sample:.2f}ms/amostra)")
    print(f"💾 Tamanho do Modelo: {model_size_kb:.1f} KB")
    print(f"✅ F1-Score (weighted): {f1:.3f}\n")
    print(report)
    
    # Guardar resultado
    results.append({
        'Model': name,
        'F1_Score': f1,
        'Train_Time_s': train_time,
        'Infer_Time_s': infer_time,
        'Infer_Time_ms_per_sample': infer_time_per_sample,
        'Model_Size_KB': model_size_kb,
        'Num_Samples_Test': len(X_test)
    })
    
    # Guardar relatório
    report_filename = f"report_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.txt"
    with open(os.path.join(METRICS_DIR, report_filename), "w") as f:
        f.write(f"Modelo: {name}\n")
        f.write(f"Tempo de Treino: {train_time:.3f}s\n")
        f.write(f"Tempo de Inferência: {infer_time:.3f}s ({infer_time_per_sample:.2f}ms/amostra)\n")
        f.write(f"Tamanho do Modelo: {model_size_kb:.1f} KB\n")
        f.write(f"F1-Score (weighted): {f1:.3f}\n\n")
        f.write(report)

# 6. COMPARAÇÃO
print("\n" + "="*100)
print("📊 COMPARAÇÃO DE MODELOS PARA EDGE COMPUTING")
print("="*100 + "\n")

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('F1_Score', ascending=False)

print(df_results.to_string(index=False))
print("\n")

# Análise de viabilidade em RPi5
print("🚀 ANÁLISE DE VIABILIDADE PARA RPi5:")
print("-" * 100)

baseline_infer = df_results[df_results['Model'].str.contains('SVM')]['Infer_Time_ms_per_sample'].values[0]

for idx, row in df_results.iterrows():
    speedup = baseline_infer / row['Infer_Time_ms_per_sample']
    f1_degradation = (df_results.iloc[0]['F1_Score'] - row['F1_Score']) * 100
    
    print(f"\n{row['Model']}:")
    print(f"  F1-Score: {row['F1_Score']:.3f} (degradação: {f1_degradation:+.1f}%)")
    print(f"  Inferência: {row['Infer_Time_ms_per_sample']:.2f}ms/amostra (speedup: {speedup:.1f}×)")
    print(f"  Tamanho: {row['Model_Size_KB']:.1f}KB")
    
    if row['Infer_Time_ms_per_sample'] < 5 and row['F1_Score'] > 0.8:
        print(f"  ✅ RECOMENDADO PARA RPi5!")
    elif row['Infer_Time_ms_per_sample'] < 10 and row['F1_Score'] > 0.75:
        print(f"  ⚠️  POSSÍVEL para RPi5 (depende de throughput)")
    else:
        print(f"  ❌ NÃO RECOMENDADO para RPi5")

# Guardar resultados
df_results.to_csv(os.path.join(METRICS_DIR, "edge_models_comparison.csv"), index=False)

print("\n" + "="*100)
print("✅ Teste concluído! Resultados guardados em: edge_models_comparison.csv")
print("="*100)
