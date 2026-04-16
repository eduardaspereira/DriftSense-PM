import pandas as pd
import numpy as np
import os
import joblib
import yaml
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore')

# 1. SETUP
CONFIG_PATH = "../configs/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config['paths']['processed_dir']
MODELS_DIR = config['paths']['models_dir']

model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
caminho_ref = [f for f in os.listdir(PROCESSED_DIR) if f.startswith('D0_')][0]
df_ref = pd.read_csv(os.path.join(PROCESSED_DIR, caminho_ref))
ref_temp = df_ref['Temp_Mean'].values

# --- GRELHA DE TESTE (Podes ajustar estes valores) ---
PERSISTENCE_GRID = [5, 10, 15, 20, 30]
ALPHA_KS_GRID = [0.01, 0.001, 0.0001, 1e-05, 1e-06]
WINDOW_SIZE = 20

def evaluate(csv_file, p_val_thresh, persist_val):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, csv_file))
    features = [c for c in df.columns if c not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
    
    det1_idx, det2_idx = None, None
    consecutive_alarms = 0
    window_temp = []
    
    for i in range(len(df)):
        X_curr = scaler.transform(df.iloc[[i]][features])
        y_pred = model.predict(X_curr)[0]
        temp_curr = df.iloc[i]['Temp_Mean']
        window_temp.append(temp_curr)
        if len(window_temp) > WINDOW_SIZE: window_temp.pop(0)

        # DET1 Logic
        if det1_idx is None:
            if y_pred == -1: consecutive_alarms += 1
            else: consecutive_alarms = 0
            if consecutive_alarms >= persist_val: det1_idx = i
        
        # DET2 Logic
        if det2_idx is None and len(window_temp) == WINDOW_SIZE:
            _, p = ks_2samp(ref_temp, window_temp)
            if p < p_val_thresh: det2_idx = i
            
    return det1_idx, det2_idx

# 2. LOOP DE OTIMIZAÇÃO
results = []
scenarios = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv')]

print("🔎 A iniciar Grid Search de Parâmetros... Isto pode demorar um pouco.")

for p_val in ALPHA_KS_GRID:
    for persist in PERSISTENCE_GRID:
        fps_det1, fps_det2 = 0, 0
        delays_det1, delays_det2 = [], []
        
        for csv in scenarios:
            d1, d2 = evaluate(csv, p_val, persist)
            
            if "D0" in csv:
                if d1 is not None: fps_det1 += 1
                if d2 is not None: fps_det2 += 1
            else:
                if d1 is not None: delays_det1.append(d1)
                if d2 is not None: delays_det2.append(d2)
        
        results.append({
            'Alpha_KS': p_val,
            'Persistence': persist,
            'DET1_FPR': fps_det1,
            'DET2_FPR': fps_det2,
            'DET1_AvgDelay': np.mean(delays_det1) if delays_det1 else 999,
            'DET2_AvgDelay': np.mean(delays_det2) if delays_det2 else 999
        })

# 3. RESULTADO FINAL
df_opt = pd.DataFrame(results)
# Ordenar pela melhor combinação para o DET1 (Zero Falsos Positivos e menor atraso)
df_opt = df_opt.sort_values(by=['DET1_FPR', 'DET1_AvgDelay'])

print("\n🏆 TOP 5 Melhores Configurações (Ordenado por Performance DET1):")
print(df_opt.head(5).to_string(index=False))

# Guardar para o relatório
df_opt.to_csv("../results/metrics/optimization_results.csv", index=False)