"""
Descrição: Extração de features temporais e de frequência a partir dos dados raw.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Script responsável por transformar ficheiros raw em janelas de features
utilizáveis pelos modelos de deteção. Produz ficheiros CSV no diretório
`processed` definidos em `configs/driftsense_dataset/config.yaml`.
"""

import os
import yaml
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. CARREGAR CONFIGURAÇÃO
CONFIG_PATH = "../../configs/driftsense_dataset/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

RAW_DIR = config['paths']['raw_data_dir']
PROCESSED_DIR = config['paths']['processed_dir']
TAMANHO_JANELA = config['feature_engineering']['window_size']
TAXA_AMOSTRAGEM = config['system']['sampling_rate_hz']
PASSO = config['feature_engineering']['step_size']

os.makedirs(PROCESSED_DIR, exist_ok=True)
colunas_corretas = ['Timestamp', 'Scenario', 'Temp', 'Hum', 'AccX', 'AccY', 'AccZ', 'SysState', 'SampleCount']
colunas_vibracao = ['AccX', 'AccY', 'AccZ']

def calcular_frequencia_pico(dados, fs):
    """Calcular a frequência de pico de um sinal.

    Retorna 0.0 se a janela for inválida ou constante.
    """
    n = len(dados)
    if n == 0 or np.all(dados == dados[0]):
        return 0.0
    yf = np.abs(rfft(dados))
    xf = rfftfreq(n, 1 / fs)
    idx_pico = np.argmax(yf[1:]) + 1
    return round(xf[idx_pico], 3)

def main():
    print("A iniciar o Pipeline de Feature Engineering (ACM Compliant)...")
    ficheiros_raw = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv') and f.startswith('D')]

    for ficheiro in ficheiros_raw:
        caminho_entrada = os.path.join(RAW_DIR, ficheiro)
        caminho_saida = os.path.join(PROCESSED_DIR, ficheiro.replace('.csv', '_features.csv'))

        print(f"\nA processar o ficheiro: {ficheiro}...")

        try:
            df_bruto = pd.read_csv(caminho_entrada, names=colunas_corretas, header=0)

            linhas_extraidas = []

            for i in range(0, len(df_bruto) - TAMANHO_JANELA + 1, PASSO):
                janela = df_bruto.iloc[i:i + TAMANHO_JANELA]

                resumo = {
                    'Scenario': janela['Scenario'].iloc[0],
                    'Temp_Mean': round(janela['Temp'].astype(float).mean(), 2),
                    'Hum_Mean': round(janela['Hum'].astype(float).mean(), 2)
                }

                for eixo in colunas_vibracao:
                    dados_eixo = janela[eixo].astype(float).values
                    std_val = np.std(dados_eixo)

                    resumo[f'{eixo}_Mean'] = round(np.mean(dados_eixo), 2)
                    resumo[f'{eixo}_Std']  = round(std_val, 2)
                    resumo[f'{eixo}_Max']  = round(np.max(dados_eixo), 2)
                    resumo[f'{eixo}_Min']  = round(np.min(dados_eixo), 2)
                    resumo[f'{eixo}_RMS']  = round(np.sqrt(np.mean(dados_eixo**2)), 2)

                    if std_val < 0.0001:
                        resumo[f'{eixo}_Skew'], resumo[f'{eixo}_Kurt'] = 0.0, 0.0
                    else:
                        resumo[f'{eixo}_Skew'] = round(skew(dados_eixo), 3)
                        resumo[f'{eixo}_Kurt'] = round(kurtosis(dados_eixo), 3)

                    resumo[f'{eixo}_PeakFreq_Hz'] = calcular_frequencia_pico(dados_eixo, TAXA_AMOSTRAGEM)

                linhas_extraidas.append(resumo)

            df_features = pd.DataFrame(linhas_extraidas)
            df_features.to_csv(caminho_saida, index=False)
            print(f"Sucesso em {ficheiro}: {len(df_features)} janelas extraídas.")

        except Exception as e:
            print(f"Erro a processar {ficheiro}: {e}")

if __name__ == "__main__":
    main()