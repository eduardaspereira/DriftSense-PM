import pandas as pd
import numpy as np

# 1. As colunas corretas (atualizadas para o nosso formato)
colunas_corretas = ['Timestamp', 'Scenario', 'Temp', 'Hum', 'AccX', 'AccY', 'AccZ', 'SysState', 'SampleCount']

print("A ler D0_dataset.csv...")
df_d3 = pd.read_csv('D0_dataset.csv', names=colunas_corretas, header=0)

# 2. Definições do Ruído e do Bias
fator_de_ruido = 0.15 
bias_offset = 20.0  # Simula a perda de calibração do sensor com o tempo
np.random.seed(42)  # Mantido para reprodutibilidade (excelente prática)

colunas_vibracao = ['AccX', 'AccY', 'AccZ']

for eixo in colunas_vibracao:
    # Calcula o ruído gaussiano
    sigma = df_d3[eixo].std() * fator_de_ruido
    ruido_gaussiano = np.random.normal(0, sigma, size=len(df_d3))
    
    # Soma o ruído e o bias, arredondando a 1 casa decimal
    df_d3[eixo] = (df_d3[eixo] + ruido_gaussiano + bias_offset).round(1)

# 3. Atualizar a Label estritamente para D3
df_d3['Scenario'] = 'D3'

# 4. Guardar o novo dataset
df_d3.to_csv('D3_dataset.csv', index=False)

print("Cenário D3 gerado com sucesso!")