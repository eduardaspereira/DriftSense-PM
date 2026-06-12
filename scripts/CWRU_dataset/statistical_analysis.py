#!/usr/bin/env python3
"""
DriftSense-PM: Gerador de Análises Estatísticas e Gráficos (Artigo)
Lê o ficheiro fatorial gerado pelo master_script.py e exporta figuras e tabelas de resumo.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import sys
import yaml

# Configurações de Estilo para o Artigo (IEEE / ACM)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# --- CARREGAR CONFIGURAÇÃO DINAMICAMENTE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/CWRU_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Erro: Ficheiro de configuração não encontrado em: {CONFIG_PATH}")
    sys.exit(1)

RESULTS_DIR = get_abs_path(config['paths']['results_dir'])
FIGURES_DIR = get_abs_path(config['paths']['figures_dir'])

def find_latest_results_file():
    """Encontra o CSV mais recente gerado pelo master_script na pasta correta de acordo com as configurações."""
    print(f"A procurar ficheiros CSV em: {RESULTS_DIR}")
    
    list_of_files = glob.glob(os.path.join(RESULTS_DIR, 'factorial_matrix_*.csv'))
    if not list_of_files:
        print(f"ERRO: Não encontrei nenhum ficheiro 'factorial_matrix_*.csv' em {RESULTS_DIR}")
        print("Corre o master_script.py primeiro para gerar os dados.")
        sys.exit(1)
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Ficheiro selecionado: {os.path.basename(latest_file)}")
    return latest_file

def calculate_confidence_interval(data, confidence=0.95):
    """Calcula a média e a margem de erro (IC) de uma lista de valores."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def plot_delay_and_far(df, figures_dir):
    """Gera o gráfico de barras comparativo de Atraso e FAR (Secção VI.A)."""
    print("\n--- A gerar Gráfico: Atraso e Taxa de Falsos Alarmes (FAR) ---")
    
    # Filtrar apenas os cenários com Drift (ignorar D0_Baseline) e manter A0_None (pois a deteção é independente da adaptação)
    df_drift = df[(df['Scenario'] != 'D0_Baseline') & (df['Adaptation'] == 'A0_None') & (df['Detector'] != 'DET0_None')]
    
    if df_drift.empty:
        print("Aviso: Não há dados suficientes para desenhar o gráfico de Deteção.")
        return

    # Tratar os "N/D" (Não Detetou) convertendo para um número muito alto para o gráfico, ou removendo
    df_drift['Delay_Windows'] = pd.to_numeric(df_drift['Delay_Windows'], errors='coerce')
    
    # Agregar dados por Detetor (Média e IC95%)
    summary = df_drift.groupby('Detector').agg(
        Delay_Mean=('Delay_Windows', 'mean'),
        Delay_Std=('Delay_Windows', lambda x: calculate_confidence_interval(x.dropna())[1]),
        FAR_Mean=('FAR_%', 'mean')
    ).reset_index()

    # Formatar nomes
    summary['Detector'] = summary['Detector'].str.replace('DET1_OCSVM', 'OC-SVM (DET1)')\
                                             .str.replace('DET2_KS', 'KS Test (DET2)')\
                                             .str.replace('DET3_ADWIN', 'ADWIN (DET3)')

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Eixo 1: Atraso (Barras)
    x = np.arange(len(summary['Detector']))
    width = 0.4
    
    bars1 = ax1.bar(x - width/2, summary['Delay_Mean'], width, yerr=summary['Delay_Std'], 
                    capsize=5, label='Atraso Médio (Janelas)', color='#2c3e50', edgecolor='black')
    
    ax1.set_ylabel('Atraso na Deteção (Janelas)', color='#2c3e50', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary['Detector'])
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    
    # Eixo 2: FAR (Linha/Pontos)
    ax2 = ax1.twinx()
    line2 = ax2.plot(x + width/2, summary['FAR_Mean'], color='#e74c3c', marker='o', markersize=8, 
                     linestyle='dashed', linewidth=2, label='Taxa Falsos Alarmes (FAR %)')
    
    ax2.set_ylabel('FAR (%)', color='#e74c3c', fontweight='bold')
    ax2.set_ylim(-0.5, max(summary['FAR_Mean']) * 1.5 if max(summary['FAR_Mean']) > 0 else 5)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    plt.title('Desempenho dos Detetores: Atraso vs FAR (Todos os Cenários)')
    
    # Combinar legendas
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.savefig(os.path.join(figures_dir, 'fig_drift_detection.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'fig_drift_detection.png'), bbox_inches='tight')
    print(f"✓ Guardado: {os.path.join(figures_dir, 'fig_drift_detection.pdf')}")

def plot_hardware_costs(df, figures_dir):
    """Gera o gráfico de barras dos tempos de Adaptação com Barras de Erro (Secção VI.C)."""
    print("\n--- A gerar Gráfico: Custos de Adaptação (Latência) ---")
    
    # Filtrar apenas os cenários com Adaptação (ignorando A0)
    df_adapt = df[df['Adaptation'] != 'A0_None']
    
    if df_adapt.empty:
        print("Aviso: Não há dados suficientes para desenhar o gráfico de Adaptação.")
        return

    # Agregar dados por Tipo de Adaptação (Média e IC95%)
    summary = df_adapt.groupby('Adaptation').agg(
        Latency_Mean=('Adapt_Latency_ms', 'mean'),
        Latency_Err=('Adapt_Latency_ms', lambda x: calculate_confidence_interval(x.dropna())[1]),
        CPU_Mean=('CPU_Peak_%', 'mean'),
        RAM_Mean=('RAM_Alloc_MB', 'mean')
    ).reset_index()

    # Formatar nomes
    summary['Adaptation'] = summary['Adaptation'].str.replace('A1_Global', 'A1 (Global)')\
                                                 .str.replace('A2_Incremental', 'A2 (Incremental)')

    fig, ax = plt.subplots(figsize=(6, 5))
    
    x = np.arange(len(summary['Adaptation']))
    width = 0.5
    
    bars = ax.bar(x, summary['Latency_Mean'], width, yerr=summary['Latency_Err'], 
                  capsize=5, color=['#e67e22', '#27ae60'], edgecolor='black')
    
    ax.set_ylabel('Latência de Recalibragem (ms)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Adaptation'])
    ax.set_title('Custo Temporal de Adaptação na Edge (Raspberry Pi 5)')
    
    # Adicionar os valores numéricos no topo das barras
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(summary['Latency_Mean']) * 0.05), 
                f'{yval:.2f} ms', ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(figures_dir, 'fig_hardware_latency.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'fig_hardware_latency.png'), bbox_inches='tight')
    print(f"✓ Guardado: {os.path.join(figures_dir, 'fig_hardware_latency.pdf')}")

def print_statistical_summary(df):
    """Imprime na consola os resumos estatísticos prontos a copiar para o LaTeX."""
    print("\n" + "="*80)
    print(" RESUMO ESTATÍSTICO PARA O ARTIGO (COPIAR PARA O LATEX)")
    print("="*80)

    # 1. Baseline Stability (D0)
    d0_data = df[(df['Scenario'] == 'D0_Baseline') & (df['Detector'] == 'DET1_OCSVM')]
    if not d0_data.empty:
        far_d0 = d0_data['FAR_%'].mean()
        print(f"\n[Secção VI.A] Estabilidade da Baseline (D0):")
        print(f"O modelo OC-SVM (DET1) registou uma FAR média de {far_d0:.2f}% no cenário de controlo.")

    # 2. Resumo dos Detetores
    print(f"\n[Secção VI.A] Resumo do Atraso de Deteção (Média ± IC95%):")
    df_drift = df[(df['Scenario'] != 'D0_Baseline') & (df['Adaptation'] == 'A0_None')]
    df_drift['Delay_Windows'] = pd.to_numeric(df_drift['Delay_Windows'], errors='coerce')
    
    for det in df_drift['Detector'].unique():
        data_det = df_drift[df_drift['Detector'] == det]['Delay_Windows'].dropna()
        if len(data_det) > 1:
            mean, err = calculate_confidence_interval(data_det)
            far_mean = df_drift[df_drift['Detector'] == det]['FAR_%'].mean()
            print(f"- {det}: Atraso = {mean:.2f} ± {err:.2f} janelas | FAR = {far_mean:.2f}%")

    # 3. Resumo de Hardware
    print(f"\n[Secção VI.C] Custos Termoelétricos (Média ± IC95%):")
    df_adapt = df[df['Adaptation'] != 'A0_None']
    for adapt in df_adapt['Adaptation'].unique():
        data_lat = df_adapt[df_adapt['Adaptation'] == adapt]['Adapt_Latency_ms'].dropna()
        if len(data_lat) > 1:
            mean_lat, err_lat = calculate_confidence_interval(data_lat)
            cpu_mean = df_adapt[df_adapt['Adaptation'] == adapt]['CPU_Peak_%'].mean()
            ram_mean = df_adapt[df_adapt['Adaptation'] == adapt]['RAM_Alloc_MB'].mean()
            print(f"- {adapt}: Latência = {mean_lat:.2f} ± {err_lat:.2f} ms | CPU = {cpu_mean:.2f}% | RAM = {ram_mean:.2f} MB")

    print("="*80 + "\n")

def main():
    # Encontrar ficheiro na pasta correta da configuração
    csv_file = find_latest_results_file()
    
    # Garantir que a diretoria de figuras configurada existe
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Ler dados
    df = pd.read_csv(csv_file)
    print(f"Dados carregados com sucesso: {len(df)} registos analisados.")
    
    # Gerar análises utilizando a pasta do config.yaml
    plot_delay_and_far(df, FIGURES_DIR)
    plot_hardware_costs(df, FIGURES_DIR)
    print_statistical_summary(df)

if __name__ == "__main__":
    main()