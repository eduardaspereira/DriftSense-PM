#!/usr/bin/env python3
"""
Wrapper para executar master_script.py com diferentes adaptações
Modifica config.yaml temporariamente e executa testes separados
"""

import subprocess
import yaml
import sys
import os
from pathlib import Path

CONFIG_FILE = '../configs/config.yaml'

def modify_config_for_adaptation(adaptation_type):
    """Modificar config.yaml para executar apenas uma adaptação específica"""
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # Guardar configuração original
    original_config = config.copy()
    
    # Criar versão modificada que força apenas uma adaptação
    config['experiment']['forced_adaptation'] = adaptation_type
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)
    
    return original_config

def restore_config(original_config):
    """Restaurar config original"""
    # Remover campo adicionado
    if 'forced_adaptation' in original_config.get('experiment', {}):
        del original_config['experiment']['forced_adaptation']
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(original_config, f)

def run_test(adaptation, repetitions):
    """Executar master_script para uma adaptação específica"""
    print(f"\n{'='*70}")
    print(f"🔄 Executando testes com Adaptação: {adaptation}")
    print(f"{'='*70}\n")
    
    original = modify_config_for_adaptation(adaptation)
    
    try:
        result = subprocess.run(
            [sys.executable, 'master_script.py', '--repetitions', str(repetitions)],
            check=True
        )
        print(f"✅ Teste {adaptation} completado com sucesso")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erro ao executar teste {adaptation}")
        return False
    finally:
        restore_config(original)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Executar testes por adaptação')
    parser.add_argument('--repetitions', type=int, default=5, 
                       help='Número de repetições (default: 5)')
    parser.add_argument('--adaptations', nargs='+', default=['A0', 'A1', 'A2'],
                       help='Adaptações a testar (default: A0 A1 A2)')
    parser.add_argument('--skip-consolidation', action='store_true',
                       help='Não consolidar resultados em ficheiro único')
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(__file__) or '.')
    
    print(f"\n🚀 Iniciando testes por Adaptação")
    print(f"   Repetições: {args.repetitions}")
    print(f"   Adaptações: {', '.join(args.adaptations)}")
    
    results_files = []
    
    for adaptation in args.adaptations:
        if adaptation not in ['A0', 'A1', 'A2']:
            print(f"⚠️  Adaptação desconhecida: {adaptation}")
            continue
        
        success = run_test(adaptation, args.repetitions)
        if success:
            results_files.append((adaptation, '../results/metrics/full_factorial_results.csv'))
    
    # Consolidar resultados (opcional)
    if not args.skip_consolidation and results_files:
        consolidate_results(results_files)

def consolidate_results(results_files):
    """Consolidar resultados de múltiplos testes em ficheiro único"""
    import pandas as pd
    
    all_results = []
    
    for adaptation, filepath in results_files:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['TestGroup'] = adaptation
            all_results.append(df)
            print(f"✅ Consolidado: {adaptation} ({len(df)} linhas)")
    
    if all_results:
        consolidated = pd.concat(all_results, ignore_index=True)
        output = '../results/metrics/consolidated_adaptation_comparison.csv'
        consolidated.to_csv(output, index=False)
        print(f"\n📊 Consolidado em: {output}")
        print(f"   Total de linhas: {len(consolidated)}")

if __name__ == '__main__':
    main()
