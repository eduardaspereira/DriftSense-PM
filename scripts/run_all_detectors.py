import pandas as pd
import numpy as np
import os
import joblib
import yaml
from sklearn.metrics import f1_score
from scipy.stats import ks_2samp
import warnings

# Silenciar avisos para o terminal ficar limpo
warnings.filterwarnings('ignore')

# --- 1. CARREGAR CONFIGURAÇÕES (Obrigatório pela ACM/Plano) ---
try:
    with open('../configs/experiment_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("❌ Erro: Ficheiro '../configs/experiment_config.yaml' não encontrado.")
    print("Por favor, cria o ficheiro e as pastas conforme o plano.")
    exit()

# Extrair os parâmetros do YAML
PROCESSED_DIR = config['paths']['processed_dir']
MODELS_DIR = config['paths']['models_dir']
RESULTS_DIR = config['paths']['results_dir']
os.makedirs(RESULTS_DIR, exist_ok=True)

REPETITIONS = config['experiment']['repetitions']
WINDOW_SIZE = config['experiment']['window_size']
F1_THRESHOLD = config['detectors']['det1']['f1_threshold']
PERSISTENCE = config['detectors']['det1']['persistence']
ALPHA_KS = config['detectors']['det2']['alpha_ks']

# Mapeamento oficial
LABEL_MAP = {
    'D0': 'NORMAL', 'D1': 'TEMP_FAULT', 
    'D2': 'RPM_FAULT', 'D3': 'NOISE_FAULT', 'D4': 'DRIFT_MIX'
}

# --- 2. CARREGAMENTO DE ARTEFACTOS ---
print("📂 A carregar modelo, encoder e dados de referência...")
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'baseline_model.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    df_ref = pd.read_csv(os.path.join(PROCESSED_DIR, 'D0_dataset_features.csv'))
except FileNotFoundError as e:
    print(f"❌ Erro: Ficheiros base não encontrados.\n{e}")
    exit()

def simulate_stream(file_name, detector_type):
    """Simula o fluxo de dados janela a janela."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, file_name))
    prefix = file_name.split('_')[0]
    target_label = LABEL_MAP.get(prefix, 'UNKNOWN')
    
    if 'label' not in df.columns:
        df['label'] = target_label

    features_cols = [c for c in df.columns if c not in ['Scenario', 'label']]
    
    detection_idx = None
    consecutive_low_f1 = 0
    preds = []
    reals = []
    
    for i in range(len(df)):
        X_curr = df.iloc[[i]][features_cols]
        y_real_str = df.iloc[i]['label']
        
        try:
            y_real_num = le.transform([y_real_str])[0]
        except:
            y_real_num = -1 

        y_pred = model.predict(X_curr)[0]
        preds.append(y_pred)
        reals.append(y_real_num)

        if detector_type == 'DET0':
            continue
            
        elif detector_type == 'DET1' and detection_idx is None:
            if i >= WINDOW_SIZE:
                current_f1 = f1_score(reals[-WINDOW_SIZE:], preds[-WINDOW_SIZE:], average='weighted')
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

    final_f1 = f1_score(reals, preds, average='weighted')
    return detection_idx, final_f1

# --- 3. EXECUÇÃO DA CAMPANHA (Com Repetições Estatísticas) ---
scenarios = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv') and 'features' in f]
detectors = ['DET0', 'DET1', 'DET2']
results = []

print(f"🔬 A iniciar simulação ({REPETITIONS} repetições) em {len(scenarios)} ficheiros...")

for csv in sorted(scenarios):
    for det in detectors:
        delays_array = []
        f1_array = []
        
        # Loop obrigatório para validação estatística
        for rep in range(REPETITIONS):
            idx, f1 = simulate_stream(csv, det)
            delays_array.append(idx if idx is not None else np.nan)
            f1_array.append(f1)
            
        # Limpar NaNs para cálculo matemático
        valid_delays = [d for d in delays_array if not pd.isna(d)]
        
        # Cálculos de Média e Desvio Padrão
        mean_delay = np.mean(valid_delays) if valid_delays else "N/D"
        std_delay = np.std(valid_delays) if valid_delays else 0.0
        mean_f1 = np.mean(f1_array)
        
        is_fp = 1 if ('D0' in csv and valid_delays) else 0
        
        scenario_name = csv.split('_dataset')[0]
        
        # Guardar log consolidado
        results.append({
            'Scenario': scenario_name,
            'Detector': det,
            'Delay_Mean': round(mean_delay, 2) if isinstance(mean_delay, float) else mean_delay,
            'Delay_Std': round(std_delay, 2),
            'Final_F1': round(mean_f1, 3),
            'False_Positive': is_fp
        })

# --- 4. EXIBIÇÃO E EXPORTAÇÃO ---
df_res = pd.DataFrame(results)
print("\n" + "="*70)
print("📊 RESULTADOS DA MONITORIZAÇÃO DE DRIFT (Média de 5 Repetições)")
print("="*70)
print(df_res.to_string(index=False))

output_path = os.path.join(RESULTS_DIR, 'drift_results_statistical.csv')
df_res.to_csv(output_path, index=False)
print(f"\n✅ Resultados estatísticos guardados em: {output_path}")