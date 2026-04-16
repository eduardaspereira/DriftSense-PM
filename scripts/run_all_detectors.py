import pandas as pd
import numpy as np
import os
import joblib
import yaml
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore')

# --- 1. CARREGAR CONFIGURAÇÕES ---
try:
    with open('../configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("❌ Erro: Ficheiro '../configs/config.yaml' não encontrado.")
    exit()

PROCESSED_DIR = config['paths']['processed_dir']
MODELS_DIR = config['paths']['models_dir']
RESULTS_DIR = config['paths']['results_dir']
os.makedirs(RESULTS_DIR, exist_ok=True)

REPETITIONS = config['experiment']['repetitions']
WINDOW_SIZE = config['feature_engineering']['window_size']
# Leitura direta (verifica se os caminhos no teu yaml estão exatamente assim)
PERSISTENCE = config.get('detectors', {}).get('det1_error_monitoring', {}).get('persistence', 20)
ALPHA_KS = config.get('detectors', {}).get('det2_distribution_test', {}).get('alpha_ks', 0.01)

print(f"⚙️ Parâmetros Carregados: Persistence={PERSISTENCE}, Alpha_KS={ALPHA_KS}, Window={WINDOW_SIZE}")

# --- 2. CARREGAMENTO DE ARTEFACTOS (Com Scaler!) ---
print("📂 A carregar Modelo Vencedor (LOF) e Scaler...")
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    # Referencial D0 para o DET2
    caminho_ref = [f for f in os.listdir(PROCESSED_DIR) if f.startswith('D0_')][0]
    df_ref = pd.read_csv(os.path.join(PROCESSED_DIR, caminho_ref))
    ref_temperatura = df_ref['Temp_Mean'].values 
except FileNotFoundError as e:
    print(f"❌ Erro: Ficheiros base não encontrados.\n{e}")
    exit()

def simulate_stream(file_name, detector_type):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, file_name))
    features_cols = [c for c in df.columns if c not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
    
    detection_idx = None
    consecutive_alarms = 0
    window_data_temp = []
    
    for i in range(len(df)):
        # 1. Extrair os dados da janela atual
        X_curr = df.iloc[[i]][features_cols]
        temp_curr = df.iloc[i]['Temp_Mean']
        
        # 2. NORMALIZAR ANTES DE PREVER! (Isto estava a faltar)
        X_curr_scaled = scaler.transform(X_curr)
        
        # O modelo cospe 1 (Normal) ou -1 (Anomalia)
        y_pred = model.predict(X_curr_scaled)[0]
        
        # Atualizar a memória de curto prazo (para o DET2)
        window_data_temp.append(temp_curr)
        if len(window_data_temp) > WINDOW_SIZE:
            window_data_temp.pop(0)

        # ---------------- DETECTORES ----------------
        if detector_type == 'DET0':
            continue 
            
        elif detector_type == 'DET1' and detection_idx is None:
            if y_pred == -1:
                consecutive_alarms += 1
            else:
                consecutive_alarms = 0
                
            if consecutive_alarms >= PERSISTENCE:
                detection_idx = i
                
        elif detector_type == 'DET2' and detection_idx is None:
            if len(window_data_temp) == WINDOW_SIZE:
                _, p_val = ks_2samp(ref_temperatura, window_data_temp)
                if p_val < ALPHA_KS:
                    detection_idx = i

    return detection_idx

# --- 3. EXECUÇÃO ---
scenarios = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv') and not f.startswith('D0')] 
scenarios.insert(0, caminho_ref) # Testar D0 no início

detectors = ['DET0', 'DET1', 'DET2']
results = []

print(f"🔬 A iniciar simulação de Drift ({REPETITIONS} repetições)...")

for csv in sorted(scenarios):
    scenario_name = csv.split('_dataset')[0]
    
    for det in detectors:
        delays_array = []
        for rep in range(REPETITIONS):
            idx = simulate_stream(csv, det)
            delays_array.append(idx if idx is not None else np.nan)
            
        valid_delays = [d for d in delays_array if not pd.isna(d)]
        mean_delay = np.mean(valid_delays) if valid_delays else "No Detection"
        std_delay = np.std(valid_delays) if valid_delays else 0.0
        
        is_fp = 1 if ('D0' in scenario_name and valid_delays) else 0
        
        results.append({
            'Scenario': scenario_name,
            'Detector': det,
            'Delay_Mean': round(mean_delay, 2) if isinstance(mean_delay, float) else mean_delay,
            'Delay_Std': round(std_delay, 2),
            'False_Positive': is_fp
        })

# --- 4. EXIBIÇÃO ---
df_res = pd.DataFrame(results)
print("\n" + "="*70)
print("📊 RESULTADOS DA DETEÇÃO DE DRIFT (Média de 5 Repetições)")
print("="*70)
print(df_res.to_string(index=False))

output_path = os.path.join(RESULTS_DIR, 'drift_results_statistical.csv')
df_res.to_csv(output_path, index=False)