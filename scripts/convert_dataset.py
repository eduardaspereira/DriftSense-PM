import os
import glob
import scipy.io
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import warnings

# Suprimir avisos de divisões por zero em janelas estáticas
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =====================================================================
# CONFIGURAÇÕES DO PIPELINE DE EXTRAÇÃO
# =====================================================================
TAMANHO_JANELA = 200
PASSO = 50
TAXA_AMOSTRAGEM = 48000  # Padrão do CWRU 48k

# Definição das pastas
PASTA_ENTRADA = os.path.join("CWRU", "raw")
PASTA_SAIDA = os.path.join("CWRU", "processed")

# Criar a pasta de destino automaticamente se não existir
if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)


def encontrar_sinal_de(mat_data, nome_ficheiro):
    """
    Caça dinamicamente o array de vibração do Drive End (DE).
    """
    chaves_de = [k for k in mat_data.keys() if 'DE_time' in k]
    
    if len(chaves_de) > 0:
        chave_correta = chaves_de[0]
        return mat_data[chave_correta].flatten()
    
    return None


def processar_ficheiro(caminho_mat):
    nome_ficheiro = os.path.basename(caminho_mat)
    nome_base = nome_ficheiro.replace('.mat', '')
    
    try:
        mat = scipy.io.loadmat(caminho_mat)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o ficheiro {nome_ficheiro}: {e}")
        return
        
    sinal_bruto = encontrar_sinal_de(mat, nome_ficheiro)
    
    if sinal_bruto is None:
        print(f"[AVISO] Sinal 'DE_time' não detetado em {nome_ficheiro}. Ficheiro ignorado.")
        return

    # Etiquetar semanticamente a classe de anomalia
    if 'Normal' in nome_ficheiro or 'normal' in nome_ficheiro.lower():
        cenario = 'Normal'
    else:
        cenario = 'Anomalia'

    linhas_extraidas = []
    
    # =====================================================================
    # MOTOR DE JANELAMENTO E EXTRAÇÃO MATEMÁTICA
    # =====================================================================
    for i in range(0, len(sinal_bruto) - TAMANHO_JANELA + 1, PASSO):
        janela = sinal_bruto[i:i + TAMANHO_JANELA]
        std_val = np.std(janela)
        
        # Calcular Transformada Rápida de Fourier (FFT) para a Frequência de Pico
        n = len(janela)
        yf = np.abs(rfft(janela))
        xf = rfftfreq(n, 1 / TAXA_AMOSTRAGEM)
        idx_pico = np.argmax(yf[1:]) + 1
        freq_pico = round(xf[idx_pico], 3)
        
        # Extração estrita das características pedidas (com ordenação otimizada)
        resumo = {
            'Scenario': cenario,
            'AccX_Mean': round(np.mean(janela), 4),
            'AccX_RMS': round(np.sqrt(np.mean(janela**2)), 4),
            'AccX_Skew': round(skew(janela), 4) if std_val > 0.0001 else 0.0,
            'AccX_Kurt': round(kurtosis(janela), 4) if std_val > 0.0001 else 0.0,
            'AccX_PeakFreq_Hz': freq_pico
        }
        linhas_extraidas.append(resumo)
        
    # Guardar apenas as colunas estruturadas
    df_final = pd.DataFrame(linhas_extraidas)
    caminho_saida = os.path.join(PASTA_SAIDA, f"{nome_base}_features.csv")
    df_final.to_csv(caminho_saida, index=False)
    
    print(f"[OK] {nome_ficheiro:<25} -> {len(df_final)} janelas filtradas em {PASTA_SAIDA}")


def main():
    caminho_busca = os.path.join(PASTA_ENTRADA, '*.mat')
    ficheiros_mat = glob.glob(caminho_busca)
    
    if len(ficheiros_mat) == 0:
        print(f"Nenhum ficheiro .mat encontrado na pasta '{PASTA_ENTRADA}'.")
        return
        
    print(f"=== EXTRAÇÃO DRIFTSENSE-PM (APENAS FEATURES SELECIONADAS) ===")
    print(f"Origem: {PASTA_ENTRADA}")
    print(f"Destino: {PASTA_SAIDA}\n")
    
    for f in sorted(ficheiros_mat):
        processar_ficheiro(f)
        
    print(f"\n=== SUCESSO: CSVs limpos criados na pasta '{PASTA_SAIDA}' ===")


if __name__ == '__main__':
    main()