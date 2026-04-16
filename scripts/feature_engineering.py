import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import warnings

# Silenciar avisos chatos de runtime (já que vamos tratar o erro manualmente)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. Configuração de Pastas
RAW_DIR = '../data/raw/'
PROCESSED_DIR = '../data/processed/'
TAMANHO_JANELA = 20  
TAXA_AMOSTRAGEM = 2.0 
PASSO = 2 

os.makedirs(PROCESSED_DIR, exist_ok=True)

colunas_corretas = ['Timestamp', 'Scenario', 'Temp', 'Hum', 'AccX', 'AccY', 'AccZ', 'Sample', 'Index']
colunas_vibracao = ['AccX', 'AccY', 'AccZ']

# 2. Função para o Domínio da Frequência (FFT)
def calcular_frequencia_pico(dados, fs):
    n = len(dados)
    if n == 0 or np.all(dados == dados[0]): 
        return 0.0 
    
    yf = np.abs(rfft(dados))
    xf = rfftfreq(n, 1 / fs)
    
    idx_pico = np.argmax(yf[1:]) + 1
    return round(xf[idx_pico], 3)

# 3. Motor Principal
print("🚀 A iniciar o Pipeline de Feature Engineering (Versão Blindada)...")

ficheiros_raw = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]

for ficheiro in ficheiros_raw:
    caminho_entrada = os.path.join(RAW_DIR, ficheiro)
    nome_saida = ficheiro.replace('.csv', '_features.csv')
    caminho_saida = os.path.join(PROCESSED_DIR, nome_saida)
    
    print(f"\n📦 A processar o ficheiro: {ficheiro}...")
    
    try:
        df_bruto = pd.read_csv(caminho_entrada, names=colunas_corretas, header=0)
        df_bruto = df_bruto.drop(['Sample', 'Index'], axis=1, errors='ignore')
        
        linhas_extraidas = []
        
        for i in range(0, len(df_bruto) - TAMANHO_JANELA + 1, PASSO):
            janela = df_bruto.iloc[i:i + TAMANHO_JANELA]
            
            resumo = {
                'Scenario': janela['Scenario'].iloc[0],
                'Temp_Mean': round(janela['Temp'].mean(), 2),
                'Hum_Mean': round(janela['Hum'].mean(), 2)
            }
            
            for eixo in colunas_vibracao:
                dados_eixo = janela[eixo].values
                std_val = np.std(dados_eixo)
                
                # --- DOMÍNIO DO TEMPO ---
                resumo[f'{eixo}_Mean'] = round(np.mean(dados_eixo), 2)
                resumo[f'{eixo}_Std']  = round(std_val, 2)
                resumo[f'{eixo}_Max']  = round(np.max(dados_eixo), 2)
                resumo[f'{eixo}_Min']  = round(np.min(dados_eixo), 2)
                resumo[f'{eixo}_RMS']  = round(np.sqrt(np.mean(dados_eixo**2)), 2)
                
                # Tratamento especial para evitar o erro de Catastrophic Cancellation
                if std_val < 0.0001:
                    resumo[f'{eixo}_Skew'] = 0.0
                    resumo[f'{eixo}_Kurt'] = 0.0
                else:
                    resumo[f'{eixo}_Skew'] = round(skew(dados_eixo), 3)
                    resumo[f'{eixo}_Kurt'] = round(kurtosis(dados_eixo), 3)
                
                # --- DOMÍNIO DA FREQUÊNCIA ---
                resumo[f'{eixo}_PeakFreq_Hz'] = calcular_frequencia_pico(dados_eixo, TAXA_AMOSTRAGEM)

            linhas_extraidas.append(resumo)
        
        df_features = pd.DataFrame(linhas_extraidas)
        df_features.to_csv(caminho_saida, index=False)
        print(f"✅ Sucesso! {len(df_features)} janelas extraídas.")
        
    except Exception as e:
        print(f"❌ Erro a processar {ficheiro}: {e}")

print("\n🎉 TUDO PRONTO! Agora podes avançar para o treino do modelo sem avisos chatos.")