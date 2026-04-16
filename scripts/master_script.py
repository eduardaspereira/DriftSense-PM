import pandas as pd
import numpy as np
import os
import joblib
import yaml
import time
import warnings
from sklearn.metrics import f1_score
from scipy.stats import ks_2samp

# Importar o TEU módulo de adaptações!
import adaptations 

warnings.filterwarnings('ignore')

# --- 1. CARREGAR CONFIGURAÇÕES ---
try:
    with open('../configs/experiment_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("❌ Erro: Ficheiro config não encontrado.")
    exit()

PROCESSED_DIR = config['paths']['processed_dir']
MODELS_DIR = config['paths']['models_dir']
RESULTS_DIR = config['paths']['results_dir']
os.makedirs(RESULTS_DIR, exist_ok=True)

REPETITIONS = config['experiment']['repetitions']
WINDOW_SIZE = config['experiment']['window_size']
F1_THRESHOLD = config['detectors']['det1']['f1_threshold']
PERSISTENCE = config['detectors']['det1']['persistence']
ALPHA_KS = config['detectors']['det2']['alpha_ks']

A1_INTERVAL = config['adaptation']['a1']['retrain_interval']
BUFFER_SIZE = config['adaptation']['a2']['buffer_size']
RECOVERY_THRESHOLD = config['metrics']['recovery_threshold']

LABEL_MAP = {
    'D0': 'NORMAL', 'D1': 'TEMP_FAULT', 
    'D2': 'RPM_FAULT', 'D3': 'NOISE_FAULT', 'D4': 'DRIFT_MIX'
}

# --- 2. FUNÇÃO CORE DO SIMULADOR ---
def simulate_stream(file_name, detector_type, adaptation_type):
    """Simula o fluxo completo: Previsão -> Deteção -> Adaptação -> Recuperação"""
    
    # Carregar modelo fresco para cada simulação (para não herdar adaptações de loops anteriores)
    model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    df_ref = pd.read_csv(os.path.join(PROCESSED_DIR, 'D0_dataset_features.csv'))
    
    df = pd.read_csv(os.path.join(PROCESSED_DIR, file_name))
    prefix = file_name.split('_')[0]
    target_label = LABEL_MAP.get(prefix, 'UNKNOWN')
    
    if 'label' not in df.columns:
        df['label'] = target_label

    features_cols = [c for c in df.columns if c not in ['Scenario', 'label']]
    
    # Variáveis de Estado
    detection_idx = None
    consecutive_low_f1 = 0
    preds, reals = [], []
    
    # Variáveis para Adaptação e Buffers
    buffer_X, buffer_y = [], []
    total_latency_ms = 0.0
    adapted_once = False
    
    # Variáveis de Recuperação
    is_recovering = False
    recovery_start_idx = None
    recovery_time = None

    for i in range(len(df)):
        X_curr = df.iloc[[i]][features_cols]
        y_real_str = df.iloc[i]['label']
        
        try:
            y_real_num = le.transform([y_real_str])[0]
        except:
            # Correção do erro Fatal do LightGBM:
            # O modelo só aceita classes de 0 a 3. Para o D4, mapeamos para a 
            # última classe conhecida (3) para permitir o fine-tuning matemático.
            y_real_num = len(le.classes_) - 1 

        # Manter o buffer de dados recentes atualizado
        buffer_X.append(X_curr)
        buffer_y.append(y_real_num)
        if len(buffer_X) > BUFFER_SIZE:
            buffer_X.pop(0)
            buffer_y.pop(0)

        # Inferência
        start_inf = time.time()
        y_pred = model.predict(X_curr)[0]
        preds.append(y_pred)
        reals.append(y_real_num)
        
        # Calcular F1 da janela atual
        current_f1 = 1.0
        if i >= WINDOW_SIZE:
            current_f1 = f1_score(reals[-WINDOW_SIZE:], preds[-WINDOW_SIZE:], average='weighted')

        # --- FASE 1: DETEÇÃO ---
        if detector_type == 'DET1' and detection_idx is None:
            if current_f1 < F1_THRESHOLD:
                consecutive_low_f1 += 1
            else:
                consecutive_low_f1 = 0
            if consecutive_low_f1 >= PERSISTENCE:
                detection_idx = i
                
        elif detector_type == 'DET2' and detection_idx is None:
            feature_to_test = features_cols[0] 
            _, p_val = ks_2samp(df_ref[feature_to_test], X_curr[feature_to_test])
            if p_val < ALPHA_KS:
                detection_idx = i

        # --- FASE 2: ADAPTAÇÃO ---
        # A1: Periódico (Independente do detetor)
        if adaptation_type == 'A1' and i > 0 and i % A1_INTERVAL == 0:
            try:
                X_train_buf = pd.concat(buffer_X)
                # Só retreina se tiver pelo menos 2 classes no buffer (limitação do LightGBM)
                if len(set(buffer_y)) > 1: 
                    model, lat = adaptations.apply_a1_periodic_retrain(model, X_train_buf, buffer_y)
                    total_latency_ms += lat
            except Exception as e:
                pass # Ignora falhas de retreino por falta de dados

        # A2: Lightweight (Despoletado pelo Detetor)
        elif adaptation_type == 'A2' and detection_idx is not None and not adapted_once:
            try:
                X_train_buf = pd.concat(buffer_X)
                model, lat = adaptations.apply_a2_lightweight_adapt(model, X_train_buf, buffer_y)
                total_latency_ms += lat
                adapted_once = True
                is_recovering = True
                recovery_start_idx = i
            except Exception as e:
                pass

        # --- FASE 3: MÉTRICA DE RECUPERAÇÃO (Recovery Time) ---
        if is_recovering and recovery_time is None:
            if current_f1 >= RECOVERY_THRESHOLD:
                recovery_time = i - recovery_start_idx
                is_recovering = False

    final_f1 = f1_score(reals, preds, average='weighted')
    return detection_idx, total_latency_ms, recovery_time, final_f1

# --- 3. EXECUTAR A MATRIZ FATORIAL COMPLETA ---
scenarios = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv') and 'features' in f]
detectors = ['DET0', 'DET1', 'DET2']
adaptations_list = ['A0', 'A1', 'A2']
results = []

print(f"🔬 A iniciar Matriz Fatorial: {len(scenarios)} cenários | 3 Detetores | 3 Adaptações | {REPETITIONS} Repetições...")
total_runs = len(scenarios) * len(detectors) * len(adaptations_list) * REPETITIONS
current_run = 0

for csv in sorted(scenarios):
    scenario_name = csv.split('_dataset')[0]
    for det in detectors:
        for adapt in adaptations_list:
            
            delays, latencies, recoveries, f1s = [], [], [], []
            
            for rep in range(REPETITIONS):
                current_run += 1
                if current_run % 20 == 0:
                    print(f"⏳ Progresso: {current_run}/{total_runs} testes...")
                
                idx, lat, rec, f1 = simulate_stream(csv, det, adapt)
                
                delays.append(idx if idx is not None else np.nan)
                latencies.append(lat)
                recoveries.append(rec if rec is not None else np.nan)
                f1s.append(f1)
                
            # Limpar NaNs para médias
            v_delays = [d for d in delays if not pd.isna(d)]
            v_recs = [r for r in recoveries if not pd.isna(r)]
            
            results.append({
                'Scenario': scenario_name,
                'Detector': det,
                'Adaptation': adapt,
                'Delay_Mean': round(np.mean(v_delays), 1) if v_delays else "N/D",
                'Latency_ms': round(np.mean(latencies), 2),
                'Recovery_Time': round(np.mean(v_recs), 1) if v_recs else "Não Recuperou",
                'Final_F1': round(np.mean(f1s), 3)
            })

# --- 4. EXPORTAR RESULTADOS ---
df_res = pd.DataFrame(results)
print("\n" + "="*85)
print("📊 MATRIZ DE RESULTADOS FINAIS (Deteção + Adaptação)")
print("="*85)
print(df_res.to_string(index=False))

output_path = os.path.join(RESULTS_DIR, 'full_factorial_results.csv')
df_res.to_csv(output_path, index=False)
print(f"\n✅ Relatório Oficial guardado em: {output_path}")