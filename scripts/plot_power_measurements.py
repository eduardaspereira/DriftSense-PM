#!/usr/bin/env python3
"""
Gerar Gráficos de Medições de Consumo Energético - DriftSense-PM
=================================================================

Cria visualizações de dados de consumo para análise e paper.

Uso:
    python plot_power_measurements.py power_measurements_fnirsi.csv
    python plot_power_measurements.py power_measurements_fnirsi.csv --output_dir figures/
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import argparse


def ensure_dir(directory):
    """Garantir que diretório existe"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def plot_power_over_time(df, output_dir):
    """Gráfico: Potência vs Tempo"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['duration_sec']/3600, df['power_w'], 
            linewidth=1.5, color='#2E86AB', alpha=0.8, label='Potência instantânea')
    
    # Média móvel (60 pontos = ~1 min)
    window_size = min(60, len(df) // 10)
    if window_size > 1:
        ax.plot(df['duration_sec']/3600, df['power_w'].rolling(window=window_size).mean(),
                linewidth=2, color='#A23B72', label=f'Média móvel ({window_size} amostras)')
    
    ax.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Potência (W)', fontsize=12, fontweight='bold')
    ax.set_title('Potência Consumida vs Tempo - DriftSense-PM em RPi5', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'power_vs_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def plot_energy_accumulated(df, output_dir):
    """Gráfico: Energia Acumulada vs Tempo"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Converter para Wh
    energy_wh = df['energy_ws'] / 3600
    
    ax.fill_between(df['duration_sec']/3600, 0, energy_wh,
                    alpha=0.3, color='#F18F01', label='Energia acumulada')
    ax.plot(df['duration_sec']/3600, energy_wh,
            linewidth=2.5, color='#F18F01', label='Energia total')
    
    ax.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energia Acumulada (Wh)', fontsize=12, fontweight='bold')
    ax.set_title('Energia Acumulada vs Tempo - DriftSense-PM em RPi5',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'energy_accumulated.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def plot_distributions(df, output_dir):
    """Gráficos: Distribuições de Tensão, Corrente, Potência"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Tensão
    axes[0, 0].hist(df['voltage_v'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Tensão (V)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequência', fontweight='bold')
    axes[0, 0].set_title(f'Distribuição de Tensão\nMédia: {df["voltage_v"].mean():.2f}V ± {df["voltage_v"].std():.3f}V')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Corrente
    axes[0, 1].hist(df['current_a']*1000, bins=50, color='#06A77D', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Corrente (mA)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequência', fontweight='bold')
    axes[0, 1].set_title(f'Distribuição de Corrente\nMédia: {df["current_a"].mean()*1000:.2f}mA ± {df["current_a"].std()*1000:.2f}mA')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Potência
    axes[1, 0].hist(df['power_w'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Potência (W)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequência', fontweight='bold')
    axes[1, 0].set_title(f'Distribuição de Potência\nMédia: {df["power_w"].mean():.3f}W ± {df["power_w"].std():.3f}W')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Temperatura
    axes[1, 1].hist(df['temp_c'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Temperatura (°C)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequência', fontweight='bold')
    axes[1, 1].set_title(f'Distribuição de Temperatura\nMédia: {df["temp_c"].mean():.1f}°C ± {df["temp_c"].std():.2f}°C')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def plot_phase_analysis(df, output_dir):
    """Gráfico: Análise por fases de operação"""
    # Definir fases baseadas em potência
    idle_threshold = 0.5
    detection_threshold = 3.0
    
    df['phase'] = pd.cut(
        df['power_w'],
        bins=[0, idle_threshold, detection_threshold, float('inf')],
        labels=['Idle', 'Detecção', 'Retraining'],
        include_lowest=True
    )
    
    # Gráfico 1: Timeline de fases
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    # Timeline com cores
    colors = {'Idle': '#90EE90', 'Detecção': '#FFD700', 'Retraining': '#FF6B6B'}
    for phase in ['Idle', 'Detecção', 'Retraining']:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            ax1.scatter(phase_data['duration_sec']/3600, phase_data['power_w'],
                       color=colors[phase], label=phase, s=20, alpha=0.6)
    
    ax1.set_ylabel('Potência (W)', fontweight='bold')
    ax1.set_title('Análise por Fases de Operação - DriftSense-PM em RPi5',
                  fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Gráfico 2: Proporção de fases
    phase_counts = df['phase'].value_counts()
    wedges, texts, autotexts = ax2.pie(
        phase_counts,
        labels=phase_counts.index,
        autopct='%1.1f%%',
        colors=[colors.get(p, '#CCCCCC') for p in phase_counts.index],
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Proporção de Tempo por Fase', fontweight='bold')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'phase_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def plot_statistics_summary(df, output_dir):
    """Gráfico: Resumo de estatísticas"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Preparar dados
    stats_data = {
        'Tensão': [df['voltage_v'].min(), df['voltage_v'].mean(), df['voltage_v'].max()],
        'Corrente\n(×100mA)': [df['current_a'].min()*100, df['current_a'].mean()*100, df['current_a'].max()*100],
        'Potência': [df['power_w'].min(), df['power_w'].mean(), df['power_w'].max()],
        'Temperatura': [df['temp_c'].min(), df['temp_c'].mean(), df['temp_c'].max()]
    }
    
    # Gráfico de barras com Min/Mean/Max
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(stats_data))
    width = 0.25
    
    mins = [v[0] for v in stats_data.values()]
    means = [v[1] for v in stats_data.values()]
    maxs = [v[2] for v in stats_data.values()]
    
    ax1.bar(x_pos - width, mins, width, label='Mín', color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos, means, width, label='Média', color='#F18F01', alpha=0.8)
    ax1.bar(x_pos + width, maxs, width, label='Máx', color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('Valor', fontweight='bold')
    ax1.set_title('Estatísticas Resumidas - Min / Média / Máx',
                  fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stats_data.keys())
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Correlação: Temperatura vs Potência
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df['power_w'], df['temp_c'], alpha=0.5, s=10, color='#FF6B6B')
    
    # Linha de tendência
    z = np.polyfit(df['power_w'], df['temp_c'], 1)
    p = np.poly1d(z)
    ax2.plot(df['power_w'].sort_values(), p(df['power_w'].sort_values()),
            "r--", linewidth=2, label=f'Tendência')
    
    ax2.set_xlabel('Potência (W)', fontweight='bold')
    ax2.set_ylabel('Temperatura (°C)', fontweight='bold')
    ax2.set_title('Correlação: Temperatura vs Potência', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Correlação: Corrente vs Potência
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df['current_a']*1000, df['power_w'], alpha=0.5, s=10, color='#FFD700')
    
    z = np.polyfit(df['current_a']*1000, df['power_w'], 1)
    p = np.poly1d(z)
    ax3.plot(sorted(df['current_a']*1000), p(sorted(df['current_a']*1000)),
            "r--", linewidth=2)
    
    ax3.set_xlabel('Corrente (mA)', fontweight='bold')
    ax3.set_ylabel('Potência (W)', fontweight='bold')
    ax3.set_title('Correlação: Corrente vs Potência', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Textbox com resumo
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    energy_ws = df['energy_ws'].iloc[-1]
    energy_wh = energy_ws / 3600
    duration_h = df['duration_sec'].max() / 3600
    
    summary_text = f"""
    RESUMO DE CONSUMO - DriftSense-PM em RPi5
    {'=' * 60}
    
    Duração Total: {duration_h:.2f} horas
    Amostras: {len(df):,}
    
    POTÊNCIA:
        Média: {df['power_w'].mean():.3f} W
        Máxima: {df['power_w'].max():.3f} W (picos durante retraining)
        Mínima: {df['power_w'].min():.3f} W (idle)
        
    ENERGIA:
        Total: {energy_wh:.3f} Wh = {energy_ws:.0f} Ws
        Custo: ~{energy_wh * 0.20:.3f}€ (a 0.20€/kWh)
        
    TEMPERATURA:
        Média: {df['temp_c'].mean():.1f}°C
        Range: {df['temp_c'].min():.1f}°C - {df['temp_c'].max():.1f}°C
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(Path(output_dir) / 'statistics_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {Path(output_dir) / 'statistics_summary.png'}")
    plt.close()


def plot_current_vs_time(df, output_dir):
    """Gráfico: Corrente vs Tempo"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['duration_sec']/3600, df['current_a']*1000,
            linewidth=1.5, color='#06A77D', alpha=0.8)
    
    window_size = min(60, len(df) // 10)
    if window_size > 1:
        ax.plot(df['duration_sec']/3600, df['current_a'].rolling(window=window_size).mean()*1000,
                linewidth=2.5, color='#A23B72', linestyle='--', label='Média móvel')
    
    ax.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Corrente (mA)', fontsize=12, fontweight='bold')
    ax.set_title('Corrente Consumida vs Tempo - DriftSense-PM em RPi5',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'current_vs_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Gerar gráficos de medições de consumo energético"
    )
    parser.add_argument(
        'csv_file',
        help='Ficheiro CSV com medições'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='results/figures',
        help='Diretório de saída para gráficos (default: results/figures)'
    )
    
    args = parser.parse_args()
    
    # Ler dados
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"❌ Ficheiro não encontrado: {args.csv_file}")
        sys.exit(1)
    
    if len(df) == 0:
        print("❌ Ficheiro CSV vazio")
        sys.exit(1)
    
    print(f"\n📊 Gerando gráficos a partir de {args.csv_file}")
    print(f"📁 Saída: {args.output_dir}")
    
    ensure_dir(args.output_dir)
    
    # Gerar gráficos
    print("\n🎨 Gerando visualizações...")
    plot_power_over_time(df, args.output_dir)
    plot_energy_accumulated(df, args.output_dir)
    plot_current_vs_time(df, args.output_dir)
    plot_distributions(df, args.output_dir)
    plot_phase_analysis(df, args.output_dir)
    plot_statistics_summary(df, args.output_dir)
    
    print(f"\n✅ Todos os gráficos foram gerados em: {args.output_dir}")


if __name__ == '__main__':
    main()
