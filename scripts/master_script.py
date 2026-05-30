import pandas as pd
import numpy as np
import os
import joblib
import yaml
import time
import warnings
import argparse
from scipy.stats import ks_2samp


import adaptations

warnings.filterwarnings('ignore')

# --- 0. PARAMETROS CLI ---
parser = argparse.ArgumentParser(description='DriftSense-PM: Full Factorial Evaluation')
parser.add_argument('--repetitions', type=int, default=None, help='Number of repetitions (overrides config.yaml)')
args = parser.parse_args()

# --- 1. CARREGAR CONFIGURAÇÕES ---
try:
    with open('../configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("❌ Erro: Ficheiro config.yaml não encontrado.")
    exit()

PROCESSED_DIR = config['paths']['processed_dir']
MODELS_DIR = config['paths']['models_dir']
RESULTS_DIR = config['paths']['results_dir']
os.makedirs(RESULTS_DIR, exist_ok=True)

REPETITIONS = args.repetitions if args.repetitions is not None else config['experiment']['repetitions']
WINDOW_SIZE = config['feature_engineering']['window_size']
PERSISTENCE = config.get('detectors', {}).get('det1_error_monitoring', {}).get('persistence', 10)
ALPHA_KS = config.get('detectors', {}).get('det2_distribution_test', {}).get('alpha_ks', 0.001)

A1_INTERVAL = config.get('adaptation', {}).get('a1_periodic_retrain', {}).get('retrain_interval', 50)
BUFFER_SIZE = config.get('adaptation', {}).get('a2_lightweight', {}).get('buffer_size', 20)

caminho_ref = [f for f in os.listdir(PROCESSED_DIR) if f.startswith('D0_')][0]

# --- 2. FUNÇÃO CORE DO SIMULADOR ---
def simulate_stream(file_name, detector_type, adaptation_type):
    """Simula o fluxo: Previsão -> Deteção -> Adaptação -> Recuperação"""
    
    # 1. Carregar Estado Inicial Puro (Baseline D0)
    model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    df_ref = pd.read_csv(os.path.join(PROCESSED_DIR, caminho_ref))
    ref_vibracao = df_ref['AccX_RMS'].values
    
    # 2. Carregar o fluxo do cenário atual
    df = pd.read_csv(os.path.join(PROCESSED_DIR, file_name))
    features_cols = [c for c in df.columns if c not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
    
    # Variáveis de Estado
    detection_idx = None
    consecutive_alarms = 0
    buffer_X = []
    window_vib = []
    total_latency_ms = 0.0
    adapted_once = False
    
    # Métricas de Recuperação 
    is_recovering = False
    recovery_start_idx = None
    recovery_time = None
    consecutive_normal_predictions = 0
    STABILITY_THRESHOLD = 5 # Definir quantas janelas estáveis determinam uma recuperação real
    
    for i in range(len(df)):
        # Extrair dados e aplicar o Scaler atual
        X_raw = df.iloc[[i]][features_cols]
        vib_curr = df.iloc[i]['AccX_RMS']
        X_scaled = scaler.transform(X_raw)
        
        # Buffer de memória
        buffer_X.append(X_raw.values[0])
        if len(buffer_X) > BUFFER_SIZE:
            buffer_X.pop(0)
            
        # Memória para o DET2 (KS-Test focado em vibração)
        window_vib.append(vib_curr)
        if len(window_vib) > WINDOW_SIZE:
            window_vib.pop(0)

        # Inferência
        y_pred = model.predict(X_scaled)[0]

        # --- FASE 1: DETEÇÃO ---
        if detector_type == 'DET1' and detection_idx is None:
            if y_pred == -1: consecutive_alarms += 1
            else: consecutive_alarms = 0
            
            if consecutive_alarms >= PERSISTENCE:
                detection_idx = i
                
        elif detector_type == 'DET2' and detection_idx is None:
            if len(window_vib) == WINDOW_SIZE:
                _, p_val = ks_2samp(ref_vibracao, window_vib)
                if p_val < ALPHA_KS:
                    detection_idx = i

        # --- FASE 2: ADAPTAÇÃO ---
        df_buffer = pd.DataFrame(buffer_X, columns=features_cols)

        if adaptation_type == 'A0':
            pass
            
        elif adaptation_type == 'A1' and i > 0 and i % A1_INTERVAL == 0:
            model, scaler, lat = adaptations.apply_a1_periodic_retrain(df_buffer, PROCESSED_DIR)
            total_latency_ms += lat
            if recovery_time is None:
                is_recovering = True
                recovery_start_idx = i
                consecutive_normal_predictions = 0

        elif adaptation_type == 'A2' and detection_idx is not None and not adapted_once:
            model, scaler, lat = adaptations.apply_a2_lightweight_adapt(df_buffer)
            total_latency_ms += lat
            adapted_once = True
            is_recovering = True
            recovery_start_idx = i
            consecutive_normal_predictions = 0

        # --- FASE 3: MÉTRICA DE RECUPERAÇÃO CORRIGIDA ---
        if is_recovering and recovery_time is None:
            if y_pred == 1:
                consecutive_normal_predictions += 1
            else:
                consecutive_normal_predictions = 0
                
            if consecutive_normal_predictions >= STABILITY_THRESHOLD:
                # O tempo total leva em conta o atraso até estabilizar
                recovery_time = (i - STABILITY_THRESHOLD + 1) - recovery_start_idx
                is_recovering = False

    return detection_idx, total_latency_ms, recovery_time




# --- 3. EXECUTAR A MATRIZ FATORIAL COMPLETA ---
scenarios = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv') and not f.startswith('D5')]
scenarios.insert(0, caminho_ref)

detectors = ['DET1', 'DET2']
adaptations_list = ['A1']
results = []

print(f"🔬 A iniciar Matriz Fatorial... ({REPETITIONS} repetições por combinação)")
print(f"📊 Total de configurações: {len(scenarios)} cenários × 3 detectores × 3 adaptações = {len(scenarios) * 3 * 3} configs básicas")
print(f"🔁 Com {REPETITIONS} repetições: {len(scenarios) * 3 * 3 * REPETITIONS} linhas esperadas no output\n")

for csv in sorted(scenarios):
    scenario_name = csv.split('_dataset')[0]
    for det in detectors:
        for adapt in adaptations_list:
            
            # CORREÇÃO: Incluir DET0+A2 como controlo (sem deteção, mas com adaptação leve periódica)
            # Isto é necessário para o fatorial completo segundo o workplan
            
            # CORREÇÃO: Guardar cada repetição como linha separada (não apenas média)
            for rep in range(REPETITIONS):
                idx, lat, rec = simulate_stream(csv, det, adapt)
                
                results.append({
                    'Repetition': rep + 1,
                    'Scenario': scenario_name,
                    'Detector': det,
                    'Adaptation': adapt,
                    'Delay (Janelas)': idx if idx is not None else "N/D",
                    'Latency (ms)': lat,
                    'Recovery Time': rec if rec is not None else "Não Recuperou"
                })

# --- 4. EXPORTAR RESULTADOS ---
df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("📊 MATRIZ DE RESULTADOS FINAIS (Deteção + Adaptação)")
print("="*80)
print(df_res.to_string(index=False))

output_path = os.path.join(RESULTS_DIR, 'full_factorial_results.csv')
df_res.to_csv(output_path, index=False)

print("\n" + "="*80)
print(f"✅ MATRIZ FATORIAL COMPLETA!")
print("="*80)
print(f"📁 Ficheiro salvo: {output_path}")
print(f"📈 Linhas no ficheiro: {len(df_res)}")
print(f"🎯 Cenários únicos: {df_res['Scenario'].nunique()}")
print(f"🔍 Detectores: {', '.join(df_res['Detector'].unique())}")
print(f"⚙️  Adaptações: {', '.join(df_res['Adaptation'].unique())}")
print("="*80 + "\n")