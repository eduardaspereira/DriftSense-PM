#!/usr/bin/env python3
"""
Análise de Medições de Consumo Energético - DriftSense-PM
==========================================================

Analisa dados recolhidos pelo power meter e gera relatório com estatísticas.

Uso:
    python analyze_power_measurements.py power_measurements_fnirsi.csv
    python analyze_power_measurements.py power_measurements_fnirsi.csv --output power_analysis_report.txt
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def analyze_power_data(csv_file, output_file=None):
    """Analisar dados de consumo energético"""
    
    # Ler CSV
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Ficheiro não encontrado: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        sys.exit(1)
    
    if len(df) == 0:
        print("❌ Ficheiro CSV vazio")
        sys.exit(1)
    
    # Validar colunas esperadas
    required_cols = ['voltage_v', 'current_a', 'power_w', 'temp_c', 'energy_ws', 'duration_sec']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"⚠️  Colunas faltando: {missing_cols}")
    
    # Análise básica
    report = []
    report.append("=" * 80)
    report.append("📊 ANÁLISE DE CONSUMO ENERGÉTICO - DRIFTSENSE-PM")
    report.append("=" * 80)
    
    report.append(f"\n📁 Ficheiro: {csv_file}")
    report.append(f"📝 Amostras: {len(df)}")
    report.append(f"⏱️  Duração: {df['duration_sec'].max():.1f}s = {df['duration_sec'].max()/3600:.2f}h")
    
    # Estatísticas de Tensão
    report.append("\n" + "-" * 80)
    report.append("⚡ TENSÃO (Volts)")
    report.append("-" * 80)
    report.append(f"  Média: {df['voltage_v'].mean():.2f} V")
    report.append(f"  Mín/Máx: {df['voltage_v'].min():.2f} V / {df['voltage_v'].max():.2f} V")
    report.append(f"  Desvio padrão: {df['voltage_v'].std():.3f} V")
    
    # Estatísticas de Corrente
    report.append("\n" + "-" * 80)
    report.append("🔌 CORRENTE (Amperes)")
    report.append("-" * 80)
    report.append(f"  Média: {df['current_a'].mean():.4f} A = {df['current_a'].mean()*1000:.2f} mA")
    report.append(f"  Mín/Máx: {df['current_a'].min():.4f} A / {df['current_a'].max():.4f} A")
    report.append(f"  Desvio padrão: {df['current_a'].std():.5f} A")
    
    # Estatísticas de Potência
    report.append("\n" + "-" * 80)
    report.append("⚡ POTÊNCIA (Watts)")
    report.append("-" * 80)
    report.append(f"  Média: {df['power_w'].mean():.3f} W")
    report.append(f"  Mín/Máx: {df['power_w'].min():.3f} W / {df['power_w'].max():.3f} W")
    report.append(f"  Desvio padrão: {df['power_w'].std():.3f} W")
    report.append(f"  Percentil 95: {df['power_w'].quantile(0.95):.3f} W")
    
    # Estatísticas de Temperatura
    report.append("\n" + "-" * 80)
    report.append("🌡️  TEMPERATURA (°C)")
    report.append("-" * 80)
    report.append(f"  Média: {df['temp_c'].mean():.1f} °C")
    report.append(f"  Mín/Máx: {df['temp_c'].min():.1f} °C / {df['temp_c'].max():.1f} °C")
    report.append(f"  Desvio padrão: {df['temp_c'].std():.2f} °C")
    
    # Estatísticas de Energia
    energy_ws = df['energy_ws'].iloc[-1] if 'energy_ws' in df.columns else 0
    energy_wh = energy_ws / 3600
    energy_kwh = energy_wh / 1000
    
    report.append("\n" + "-" * 80)
    report.append("🔋 ENERGIA")
    report.append("-" * 80)
    report.append(f"  Total consumido: {energy_ws:.2f} Ws")
    report.append(f"  Total consumido: {energy_wh:.3f} Wh")
    report.append(f"  Total consumido: {energy_kwh:.6f} kWh")
    
    # Custo de energia
    cost_per_kwh = 0.20  # €/kWh (valor referencial, ajuste conforme seu país)
    cost = energy_kwh * cost_per_kwh
    report.append(f"  Custo estimado: {cost:.4f} € (a {cost_per_kwh}€/kWh)")
    
    # Capacidade
    if 'capacity_as' in df.columns:
        capacity_as = df['capacity_as'].iloc[-1]
        capacity_ah = capacity_as / 3600
        report.append(f"\n  Capacidade: {capacity_as:.1f} As = {capacity_ah:.6f} Ah")
    
    # Análise de fases (baseado em potência)
    report.append("\n" + "-" * 80)
    report.append("🔄 ANÁLISE POR FASES (com base em potência)")
    report.append("-" * 80)
    
    # Definir limiares de fases
    idle_threshold = 0.5  # < 0.5 W
    detection_threshold = 3.0  # 0.5 - 3 W
    
    df['phase'] = pd.cut(
        df['power_w'],
        bins=[0, idle_threshold, detection_threshold, float('inf')],
        labels=['Idle', 'Detecção', 'Retraining'],
        include_lowest=True
    )
    
    for phase in ['Idle', 'Detecção', 'Retraining']:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            duration_min = (phase_data['duration_sec'].max() - phase_data['duration_sec'].min()) / 60
            power_avg = phase_data['power_w'].mean()
            energy_phase = (phase_data['power_w'].mean() * (phase_data['duration_sec'].max() - phase_data['duration_sec'].min()))
            
            report.append(f"\n  {phase}:")
            report.append(f"    Amostras: {len(phase_data)}")
            report.append(f"    Duração: {duration_min:.2f} min")
            report.append(f"    Potência média: {power_avg:.3f} W")
            report.append(f"    Energia: {energy_phase:.2f} Ws = {energy_phase/3600:.3f} Wh")
            report.append(f"    Percentagem: {100*len(phase_data)/len(df):.1f}%")
    
    # Picos de potência
    report.append("\n" + "-" * 80)
    report.append("📈 PICOS DE POTÊNCIA (top 10)")
    report.append("-" * 80)
    
    top_power = df.nlargest(10, 'power_w')[['timestamp_iso', 'power_w', 'current_a', 'temp_c']]
    for idx, (_, row) in enumerate(top_power.iterrows(), 1):
        report.append(
            f"  {idx}. {row['power_w']:.3f}W @ {row['timestamp_iso']} "
            f"(I={row['current_a']:.4f}A, T={row['temp_c']:.1f}°C)"
        )
    
    # Recomendações
    report.append("\n" + "-" * 80)
    report.append("💡 RECOMENDAÇÕES")
    report.append("-" * 80)
    
    max_temp = df['temp_c'].max()
    if max_temp > 70:
        report.append(f"  ⚠️  Temperatura máxima ({max_temp:.1f}°C) é alta. Verifique ventilação.")
    else:
        report.append(f"  ✅ Temperatura normal ({max_temp:.1f}°C)")
    
    power_variance = df['power_w'].std() / df['power_w'].mean()
    if power_variance > 0.5:
        report.append(f"  ⚠️  Alta variabilidade de potência (CV={power_variance:.2f}). Verifique se há interferências.")
    else:
        report.append(f"  ✅ Potência estável (CV={power_variance:.2f})")
    
    if df['power_w'].max() / df['power_w'].mean() > 3:
        report.append(f"  ⚠️  Picos significativos detectados. Verifique cargas variáveis.")
    
    report.append("\n" + "=" * 80)
    
    # Imprimir relatório
    report_text = "\n".join(report)
    print(report_text)
    
    # Guardar em ficheiro se solicitado
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\n✅ Relatório guardado em: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analisar medições de consumo energético"
    )
    parser.add_argument(
        'csv_file',
        help='Ficheiro CSV com medições'
    )
    parser.add_argument(
        '--output', '-o',
        help='Ficheiro de saída para relatório'
    )
    
    args = parser.parse_args()
    
    analyze_power_data(args.csv_file, args.output)


if __name__ == '__main__':
    main()
