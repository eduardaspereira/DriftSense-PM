#!/usr/bin/env python3
"""
DriftSense-PM: Master Pipeline Execution Script
Executa sequencialmente todo o fluxo de trabalho do artigo, desde a extração
de características até à geração dos PDFs e figuras finais com rigor estatístico.
"""

import os
import sys
import subprocess
import time

# Determinar os caminhos relativos ao projeto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Se o script for executado da raiz, ajusta o caminho para a pasta dos scripts
BASE_SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts", "IMS_Bearings")

# Caso já ponhas este script dentro de 'scripts/IMS_Bearings/', descomenta a linha abaixo:
# BASE_SCRIPTS_DIR = SCRIPT_DIR

# Lista ordenada de execução conforme o pipeline do artigo
PIPELINE_SCRIPTS = [
    "01_feature_engineering.py",
    "04b_synthetic_drift.py",
    "02_selecao_modelos.py",
    "03_train_baseline.py",
    "04_drift_evaluation.py",
    "05_adaptation_strategies.py",
    "06_plot_figures.py"
]

def run_pipeline():
    print("=" * 70)
    print(" DRIFTSENSE-PM: INICIAR EXECUÇÃO DO PIPELINE MASTER")
    print("=" * 70)
    
    start_global = time.perf_counter()
    python_executable = sys.executable  # Garante que usa o mesmo (venv) ativo
    
    for idx, script in enumerate(PIPELINE_SCRIPTS, start=1):
        script_path = os.path.join(BASE_SCRIPTS_DIR, script)
        
        # Salvaguarda caso o script não esteja no sítio correto
        if not os.path.exists(script_path):
            # Tenta procurar no diretório atual de execução por conveniência
            script_path = os.path.join(SCRIPT_DIR, script)
            if not os.path.exists(script_path):
                print(f"\n[ERRO CRÍTICO] Script não encontrado: {script}")
                print(f"Caminho tentado: {os.path.join(BASE_SCRIPTS_DIR, script)}")
                sys.exit(1)
                
        print(f"\n[{idx}/{len(PIPELINE_SCRIPTS)}] A executar: {script}...")
        print("-" * 50)
        
        start_step = time.perf_counter()
        
        # Executa o script filho capturando o fluxo em tempo real no terminal
        processo = subprocess.run([python_executable, script_path])
        
        end_step = time.perf_counter()
        
        # Se o script filho retornar um código de erro (exit code != 0), aborta o pipeline
        if processo.returncode != 0:
            print(f"\n[ERRO] O script '{script}' falhou com o código {processo.returncode}.")
            print("Execução em cadeia interrompida para evitar dados corrompidos.")
            sys.exit(processo.returncode)
            
        print(f"-> Concluído com sucesso em {end_step - start_step:.2f} segundos.")
        
    end_global = time.perf_counter()
    
    print("\n" + "=" * 70)
    print(" PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f" Tempo total de execução: {(end_global - start_global) / 60:.2f} minutos.")
    print("[!] Gráficos e PDFs gerados na pasta de resultados do DriftSense-PM.")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()