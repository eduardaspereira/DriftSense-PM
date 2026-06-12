# statistical_analysis.py
"""
Descrição: Análise estatística de resultados experimentais do DriftSense-PM.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon, f_oneway
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = os.path.basename(SCRIPT_DIR)
CONFIG_NAME = f"{DATASET_NAME.replace('_dataset', '')}_config.yaml"
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", CONFIG_NAME)

def get_abs_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value.lstrip('./')))

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = get_abs_path(config['paths']['results_dir'])

def load_factorial_results(filename='full_factorial_results.csv'):
    """Load raw factorial results"""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")
    
    df = pd.read_csv(path)
    print(f"Carregado: {len(df)} linhas de {filename}")
    return df

def compute_summary_statistics(df):
    """Compute Mean ± Std for each configuration"""
    
    # Convert numeric columns
    df['Delay (Janelas)'] = pd.to_numeric(df['Delay (Janelas)'], errors='coerce')
    df['Latency (ms)'] = pd.to_numeric(df['Latency (ms)'], errors='coerce')
    
    summary = df.groupby(['Scenario', 'Detector', 'Adaptation']).agg({
        'Delay (Janelas)': ['mean', 'std', 'min', 'max', 'count'],
        'Latency (ms)': ['mean', 'std'],
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    print(f"Sumário estatístico calculado para {len(summary)} grupos")
    return summary

def compute_confidence_intervals(df, confidence=0.95):
    """Compute 95% Confidence Intervals"""
    
    df['Delay (Janelas)'] = pd.to_numeric(df['Delay (Janelas)'], errors='coerce')
    
    z_score = 1.96  # For 95% CI
    
    ci_data = []
    for (scenario, detector), group in df.groupby(['Scenario', 'Detector']):
        delays = group['Delay (Janelas)'].dropna()
        
        if len(delays) > 0:
            mean_delay = delays.mean()
            std_error = delays.std() / np.sqrt(len(delays))
            margin_of_error = z_score * std_error
            
            ci_data.append({
                'Scenario': scenario,
                'Detector': detector,
                'Mean Delay': round(mean_delay, 2),
                'Std': round(delays.std(), 2),
                'CI Lower': round(mean_delay - margin_of_error, 2),
                'CI Upper': round(mean_delay + margin_of_error, 2),
                'N': len(delays)
            })
    
    ci_df = pd.DataFrame(ci_data)
    print(f"Intervalos de confiança 95% calculados para {len(ci_df)} grupos")
    return ci_df

def wilcoxon_test(df):
    """Wilcoxon signed-rank test: DET1 vs DET2"""
    
    df['Delay (Janelas)'] = pd.to_numeric(df['Delay (Janelas)'], errors='coerce')
    
    wilcoxon_results = []
    
    for scenario in df['Scenario'].unique():
        scenario_data = df[df['Scenario'] == scenario]
        
        det1_delays = scenario_data[scenario_data['Detector'] == 'DET1']['Delay (Janelas)'].dropna()
        det2_delays = scenario_data[scenario_data['Detector'] == 'DET2']['Delay (Janelas)'].dropna()
        
        if len(det1_delays) > 0 and len(det2_delays) > 0 and len(det1_delays) == len(det2_delays):
            stat, p_value = wilcoxon(det1_delays, det2_delays, alternative='two-sided')
            
            significance = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
            
            wilcoxon_results.append({
                'Scenario': scenario,
                'Comparison': 'DET1 vs DET2',
                'p_value': round(p_value, 6),
                'Significant': significance,
                'Mean DET1': round(det1_delays.mean(), 2),
                'Mean DET2': round(det2_delays.mean(), 2),
                'Difference': round(det1_delays.mean() - det2_delays.mean(), 2)
            })
    
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    print(f"Teste Wilcoxon completado para {len(wilcoxon_df)} cenários")
    return wilcoxon_df

def adaptation_comparison(df):
    """Compare adaptation strategies (A0, A1, A2)"""
    
    df['Latency (ms)'] = pd.to_numeric(df['Latency (ms)'], errors='coerce')
    
    adaptation_stats = df.groupby('Adaptation').agg({
        'Latency (ms)': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    # Flatten
    adaptation_stats.columns = ['Mean_Latency_ms', 'Std_Latency_ms', 'Min_Latency_ms', 'Max_Latency_ms', 'N']
    
    adaptation_stats = adaptation_stats.reset_index()
    
    # Add speedup comparison
    a1_latency = adaptation_stats[adaptation_stats['Adaptation'] == 'A1']['Mean_Latency_ms'].values
    a2_latency = adaptation_stats[adaptation_stats['Adaptation'] == 'A2']['Mean_Latency_ms'].values
    
    if len(a1_latency) > 0 and len(a2_latency) > 0:
        speedup = round(a1_latency[0] / a2_latency[0], 1)
        adaptation_stats['Speedup_vs_A1'] = adaptation_stats['Adaptation'].apply(
            lambda x: 1.0 if x == 'A1' else (speedup if x == 'A2' else 1.0)
        )
    
    print(f"Comparação de adaptações completada para {len(adaptation_stats)} estratégias")
    return adaptation_stats

def save_results(summary, ci, wilcoxon_df, adaptation_df):
    """Save all results to CSV"""
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    summary.to_csv(os.path.join(RESULTS_DIR, 'full_factorial_summary.csv'))
    print(f"Salvo: full_factorial_summary.csv")
    
    ci.to_csv(os.path.join(RESULTS_DIR, 'confidence_intervals.csv'), index=False)
    print(f"Salvo: confidence_intervals.csv")
    
    wilcoxon_df.to_csv(os.path.join(RESULTS_DIR, 'wilcoxon_tests.csv'), index=False)
    print(f"Salvo: wilcoxon_tests.csv")
    
    adaptation_df.to_csv(os.path.join(RESULTS_DIR, 'adaptation_comparison.csv'), index=False)
    print(f"Salvo: adaptation_comparison.csv")

def print_summary_report(wilcoxon_df, adaptation_df):
    """Print summary report"""
    
    print("\n" + "="*80)
    print("RELATÓRIO DE ANÁLISE ESTATÍSTICA - DriftSense-PM")
    print("="*80)
    
    print("\nTESTE WILCOXON (DET1 vs DET2):")
    print("-" * 80)
    print(wilcoxon_df.to_string(index=False))
    
    print("\nCOMPARAÇÃO DE ADAPTAÇÕES:")
    print("-" * 80)
    print(adaptation_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Análise estatística concluída!")
    print("="*80 + "\n")

def main():
    """Main execution"""
    
    print("Iniciando Análise Estatística...")
    print("-" * 80)
    
    # Load results
    df = load_factorial_results()
    
    # Compute statistics
    summary = compute_summary_statistics(df)
    ci = compute_confidence_intervals(df)
    wilcoxon_df = wilcoxon_test(df)
    adaptation_df = adaptation_comparison(df)
    
    # Save results
    save_results(summary, ci, wilcoxon_df, adaptation_df)
    
    # Print report
    print_summary_report(wilcoxon_df, adaptation_df)
    
    print("Análise concluída com sucesso!")

if __name__ == "__main__":
    main()