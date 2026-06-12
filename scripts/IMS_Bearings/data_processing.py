#!/usr/bin/env python3
"""
Descrição: Extração de features temporais e espectrais avançadas (IMS Bearings Dataset).
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
"""

import os
import glob
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import sys

# --- CARREGAR CONFIGURAÇÃO DINAMICAMENTE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERRO: Ficheiro de configuração não encontrado em: {CONFIG_PATH}")
    sys.exit(1)

# Extrair caminhos e configurações do YAML
RAW_DATA_DIR = get_abs_path(config['paths']['raw_data_dir'])
PROCESSED_DATA_DIR = get_abs_path(config['paths']['processed_dir'])

# Taxa de amostragem padrão do IMS Dataset (20.48 kHz)
TAXA_AMOSTRAGEM = config['system'].get('sampling_rate_hz', 20480.0) 

def extract_features_from_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[0])
    except Exception as e:
        print(f"[AVISO] Falha ao ler o ficheiro {os.path.basename(file_path)}: {e}")
        return None
        
    janela = df[0].values
    n = len(janela)
    
    # Se o ficheiro estiver corrompido ou incompleto, salta
    if n < 20480:
        return None
        
    std_val = np.std(janela)
    rms = np.sqrt(np.mean(janela**2))
    
    # --- PROCESSAMENTO ESPECTRAL (FFT COM REMOÇÃO DE COMPONENTE DC) ---
    # Subtrair a média para centrar o sinal e eliminar o pico nos 0 Hz
    sinal_centrado = janela - np.mean(janela)
    
    yf = np.abs(rfft(sinal_centrado))
    xf = rfftfreq(n, 1 / TAXA_AMOSTRAGEM)
    
    # Encontrar a frequência de pico real (ignorando o primeiro elemento espectral)
    idx_pico = np.argmax(yf[1:]) + 1
    freq_pico = round(xf[idx_pico], 3)
    
    # 1. Energia na frequência de rotação (1X: ~33.33 Hz para 2000 RPM)
    # Criamos uma banda de guarda tolerante entre 30 Hz e 36 Hz
    indices_1x = np.where((xf >= 30.0) & (xf <= 36.0))[0]
    energy_1x = round(np.sum(yf[indices_1x]**2), 4) if len(indices_1x) > 0 else 0.0
    
    # 2. Energia em Alta Frequência (Frequências acima de 5 kHz)
    indices_high = np.where(xf > 5000.0)[0]
    energy_high = round(np.sum(yf[indices_high]**2), 4) if len(indices_high) > 0 else 0.0
    
    # Extrair timestamp a partir do nome do ficheiro (Formato IMS: YYYY.MM.DD.HH.MM.SS)
    file_name = os.path.basename(file_path)
    try:
        timestamp = datetime.strptime(file_name, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        print(f"[AVISO] Formato de data inválido no ficheiro: {file_name}")
        return None
    
    return {
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": file_name,
        "AccX_Mean": round(np.mean(janela), 4),
        "AccX_RMS": round(rms, 4),
        "AccX_Skew": round(skew(janela), 4) if std_val > 0.0001 else 0.0,
        "AccX_Kurt": round(kurtosis(janela), 4) if std_val > 0.0001 else 0.0,
        "AccX_PeakFreq_Hz": freq_pico,
        "Energy_1X_Hz": energy_1x,
        "Energy_HighFreq": energy_high,
        "Crest_Factor": round(np.max(np.abs(janela)) / rms, 4) if rms != 0 else 0.0,
        "Std_Dev": round(std_val, 4),
        "Peak_to_Peak": round(np.ptp(janela), 4)
    }

def process_ims_dataset():
    search_path = os.path.join(RAW_DATA_DIR, "2004*")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        print(f"ERRO: Nenhum ficheiro correspondente a '2004*' encontrado em {RAW_DATA_DIR}")
        sys.exit(1)
        
    print(f"Foram encontrados {len(file_list)} ficheiros para processar.")
    
    features_list = []
    
    for idx, file_path in enumerate(file_list):
        features = extract_features_from_file(file_path)
        if features is not None:
            # Rotulagem semântica rigorosa para a matriz fatorial de Concept Drift
            if idx < 500:
                features["Scenario"] = "D0_Baseline"
            elif idx < 750:
                features["Scenario"] = "D1_Ligeiro"
            else:
                features["Scenario"] = "D5_Catastrofico"
                
            features_list.append(features)
            
        if idx % 100 == 0 and idx > 0:
            print(f"Processados {idx}/{len(file_list)} ficheiros...")

    # Gerar DataFrame estruturado
    final_df = pd.DataFrame(features_list)
    
    # Guardar no diretório do CWRU / IMS unificado
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "ims_bearing1_features.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\n=== SUCESSO: Dataset espectral avançado guardado em: {output_path} ===")

if __name__ == "__main__":
    process_ims_dataset()