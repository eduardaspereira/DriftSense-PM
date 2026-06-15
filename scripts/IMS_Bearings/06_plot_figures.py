#!/usr/bin/env python3
"""
Descrição: Geração automática de gráficos com injeção de barras de erro (IC95%).
Corrigido: Tratamento defensivo de chaves ausentes (KeyError) para consistência multi-run.
"""

import os
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/IMS_Bearings_config.yaml")

def get_abs_path(path_value):
    if os.path.isabs(path_value): return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = get_abs_path(config['paths']['results_dir'])
FIGURES_DIR = get_abs_path(config['paths']['figures_dir'])
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_model_selection():
    json_path = os.path.join(RESULTS_DIR, "selecao_modelos_pareto.json")
    if not os.path.exists(json_path): 
        print(f"[Aviso] Ficheiro não encontrado: {json_path}")
        return
        
    with open(json_path, 'r') as f: 
        dados = json.load(f)
    
    df = pd.DataFrame(dados)
    
    # Se o ficheiro JSON for antigo e não tiver as barras de erro, injeta zeros para não quebrar
    if 'f1_ic95' not in df.columns:
        df['f1_ic95'] = 0.0
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Anomaly Detection Models', fontweight='bold')
    ax1.set_ylabel('Mean F1-Score', color=color, fontweight='bold')
    
    # Injeção das Error Bars (yerr) baseadas no IC95 gerado pelo loop estatístico
    bars = ax1.bar(df['modelo'], df['f1_score'], yerr=df['f1_ic95'], color=color, alpha=0.7, width=0.4, capsize=6, ecolor='black', label='F1-Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.75, 1.05)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Mean Inference Latency (ms)', color=color, fontweight='bold')  
    ax2.plot(df['modelo'], df['latencia_inferencia_ms'], color=color, marker='o', linestyle='dashed', linewidth=2, markersize=8)
    ax2.set_ylim(0, max(df['latencia_inferencia_ms']) * 1.4)
    ax2.grid(False)
    
    plt.title('Statistical Edge Execution Profile (95% Confidence Intervals)', fontweight='bold', pad=15)
    fig.tight_layout()
    out_p = os.path.join(FIGURES_DIR, "fig1_model_selection_pareto.pdf")
    plt.savefig(out_p, format='pdf', bbox_inches='tight')
    print(f"Gráfico guardado: {out_p}")

def plot_adaptation_results():
    json_path = os.path.join(RESULTS_DIR, "adaptation_metrics.json")
    if not os.path.exists(json_path): 
        print(f"[Aviso] Ficheiro não encontrado: {json_path}")
        return
        
    with open(json_path, 'r') as f: 
        dados = json.load(f)
    
    # Filtrar chaves de metadados de rede pura (como o Cloud_Centric_Raw) se existirem
    estrategias = [k for k in dados.keys() if k != "Cloud_Centric_Raw"]
    
    f1_scores = [dados[k]['f1_score'] for k in estrategias]
    # Recuperação segura com fallback (get) para evitar novos KeyErrors
    f1_ic95 = [dados[k].get('f1_ic95', 0.0) for k in estrategias]
    retrains = [dados[k]['retrains'] for k in estrategias]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_f1 = 'darkgreen'
    ax1.set_xlabel('Adaptation Strategies', fontweight='bold')
    ax1.set_ylabel('Post-Drift Mean F1-Score', fontweight='bold', color=color_f1)
    
    # Injeção de bandas/barras de erro no gráfico de linhas de adaptação
    ax1.errorbar(estrategias, f1_scores, yerr=f1_ic95, fmt='-s', color=color_f1, linewidth=2.5, markersize=8, capsize=5, ecolor='black', elinewidth=1.5, label='F1-Score (IC95%)')
    ax1.tick_params(axis='y', labelcolor=color_f1)
    ax1.set_ylim(0.5, 1.05)
    
    for i, score in enumerate(f1_scores):
        ax1.text(estrategias[i], score + 0.02, f"{score:.3f}", ha='center', va='bottom', color=color_f1, fontweight='bold')

    ax2 = ax1.twinx()
    color_ret = 'tab:orange'
    ax2.set_ylabel('Mean Executed Retrains on Edge', fontweight='bold', color=color_ret)
    bars = ax2.bar(estrategias, retrains, color=color_ret, alpha=0.3, width=0.3)
    ax2.set_ylim(0, max(retrains) + 5)
    ax2.grid(False)

    # Identificar chaves exatas dinamicamente para o cálculo da anotação
    a1_key = 'A1_Periodic' if 'A1_Periodic' in estrategias else None
    a2_key = 'A2_Lightweight_Veto' if 'A2_Lightweight_Veto' in estrategias else None
    
    if a1_key and a2_key:
        idx_a1 = estrategias.index(a1_key)
        idx_a2 = estrategias.index(a2_key)
        diff_ret = dados[a1_key]['retrains'] - dados[a2_key]['retrains']
        
        ax2.annotate(
            f'{diff_ret:.1f} Retrains Avoided\n(Resource-Saving)', 
            xy=(idx_a2, retrains[idx_a2]), 
            xytext=(idx_a1 + 0.3, (retrains[idx_a1] + retrains[idx_a2]) / 2),
            arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.5),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.15, ec="orange")
        )

    plt.title('Adaptation Robustness: Statistical Framework Stability', fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    out_p = os.path.join(FIGURES_DIR, "fig2_adaptation_robustness.pdf")
    plt.savefig(out_p, format='pdf', bbox_inches='tight')
    print(f"Gráfico guardado: {out_p}")

if __name__ == "__main__":
    plot_model_selection()
    plot_adaptation_results()