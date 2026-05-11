import os
import yaml
import pandas as pd
import numpy as np
import joblib
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# CONFIGURAÇÕES
CONFIG_PATH = "../configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config['paths']['processed_dir']
METRICS_DIR = config['paths']['results_dir']

target_names = ['Anomalia/Drift (-1)', 'Normal (1)']

# CARREGAMENTO (IGUAL SVM original)
print("📂 Testando modelos ONE-CLASS para RPi5...\n")
caminho_d0 = os.path.join(PROCESSED_DIR, "D0_dataset_features.csv")
df_d0 = pd.read_csv(caminho_d0)

X_d0 = df_d0.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
y_d0 = np.ones(len(X_d0))

X_train, X_test_normal, y_train, y_test_normal = train_test_split(X_d0, y_d0, test_size=0.2, shuffle=False)

test_anomalies = []
for file in os.listdir(PROCESSED_DIR):
    if file.endswith(".csv") and not file.startswith("D0"):
        df_anom = pd.read_csv(os.path.join(PROCESSED_DIR, file))
        X_anom = df_anom.drop(['Scenario', 'Timestamp', 'SysState', 'SampleCount'], axis=1, errors='ignore')
        y_anom = np.full(len(X_anom), -1)
        test_anomalies.append((X_anom, y_anom))

X_test = pd.concat([X_test_normal] + [anom[0] for anom in test_anomalies], ignore_index=True)
y_test = np.concatenate([y_test_normal] + [anom[1] for anom in test_anomalies])

# NORMALIZAÇÃO
print("⚖️ Normalizando features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Dataset: {len(X_train)} treino + {len(X_test)} teste\n")

# DEFINIÇÃO DE MODELOS (focado em ONE-CLASS para edge)
models = {
    "One-Class SVM (Baseline)": OneClassSVM(nu=0.01, kernel="rbf", gamma='scale'),
    "One-Class SVM (Linear)": OneClassSVM(nu=0.01, kernel="linear"),
    "LOF (n_neighbors=20)": LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True),
    "LOF (n_neighbors=5) [Lighter]": LocalOutlierFactor(n_neighbors=5, contamination=0.01, novelty=True),
    "Isolation Forest": IsolationForest(n_estimators=50, contamination=0.01, random_state=42),
    "Isolation Forest (5 trees) [Lighter]": IsolationForest(n_estimators=5, contamination=0.01, random_state=42),
}

print(f"🔬 Testando {len(models)} modelos one-class...\n")
print("="*120)

results = []

for name, model in models.items():
    print(f"\n# {name}")
    print("-" * 120)
    
    # Treino
    start_train = time.time()
    model.fit(X_train_scaled)
    train_time = time.time() - start_train
    
    # Inferência
    start_infer = time.time()
    y_pred = model.predict(X_test_scaled)
    infer_time = time.time() - start_infer
    infer_time_per_sample = (infer_time / len(X_test)) * 1000
    
    # Métricas
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    
    # Tamanho
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
    joblib.dump(model, tmp_path)
    model_size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    
    print(f"  Train: {train_time:.3f}s | Infer: {infer_time_per_sample:.3f}ms/sample | Size: {model_size_kb:.1f}KB")
    print(f"  F1-Score (weighted): {f1:.3f}\n")
    print(report)
    
    results.append({
        'Model': name,
        'F1_Score': f1,
        'Train_Time_s': train_time,
        'Infer_Time_ms_per_sample': infer_time_per_sample,
        'Model_Size_KB': model_size_kb,
    })

# ANÁLISE FINAL
print("\n" + "="*120)
print("📊 RESUMO PARA EDGE COMPUTING (RPi5)")
print("="*120 + "\n")

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('F1_Score', ascending=False)

print(df_results.to_string(index=False))
print("\n")

baseline_f1 = df_results.iloc[0]['F1_Score']
baseline_infer = df_results.iloc[0]['Infer_Time_ms_per_sample']

print("🚀 RECOMENDAÇÕES PARA RPi5:")
print("-" * 120)
for idx, row in df_results.iterrows():
    f1_loss = (baseline_f1 - row['F1_Score']) * 100
    speedup = row['Infer_Time_ms_per_sample'] / baseline_infer if baseline_infer > 0 else 1
    
    status = ""
    if row['F1_Score'] > 0.85 and row['Model_Size_KB'] < 100:
        status = "✅ RECOMENDADO"
    elif row['F1_Score'] > 0.80 and row['Model_Size_KB'] < 50:
        status = "⚠️  POSSÍVEL (trade-off)"
    else:
        status = "❌ NÃO RECOMENDADO"
    
    print(f"\n{row['Model']}: {status}")
    print(f"  F1: {row['F1_Score']:.3f} (perda: {f1_loss:+.1f}%)")
    print(f"  Latência: {row['Infer_Time_ms_per_sample']:.3f}ms/amostra")
    print(f"  Tamanho: {row['Model_Size_KB']:.1f}KB")

# Guardar
df_results.to_csv(os.path.join(METRICS_DIR, "edge_models_oneclass_comparison.csv"), index=False)
print("\n✅ Resultados guardados: edge_models_oneclass_comparison.csv")
