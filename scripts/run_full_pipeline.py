#!/usr/bin/env python3
"""
DriftSense-PM: Complete Pipeline Orchestrator
==============================================

Executes full end-to-end pipeline:
Feature Extraction → Model Training → Factorial Evaluation →  
Statistical Analysis → Plot Generation

Author: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
Date: May 7, 2026
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# ANSI colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class PipelineOrchestrator:
    """Orchestrate and monitor pipeline execution"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stages = []
        self.failed_stages = []
    
    def print_header(self, text):
        """Print formatted header"""
        print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}🚀 {text}{RESET}")
        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")
    
    def print_stage(self, number, description):
        """Print stage info"""
        print(f"{BOLD}{YELLOW}[Stage {number}/5]{RESET} {description}")
        print("-" * 80)
    
    def print_success(self, message):
        """Print success message"""
        print(f"{GREEN}✅ {message}{RESET}\n")
    
    def print_error(self, message):
        """Print error message"""
        print(f"{RED}❌ {message}{RESET}\n")
    
    def run_command(self, cmd, description, stage_num):
        """Execute a command and handle errors"""
        
        self.print_stage(stage_num, description)
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=False,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                self.print_error(f"Falhou: {description}")
                self.failed_stages.append((stage_num, description))
                return False
            
            self.print_success(description)
            self.stages.append((stage_num, description, "PASSED"))
            return True
        
        except Exception as e:
            self.print_error(f"Exceção em {description}: {str(e)}")
            self.failed_stages.append((stage_num, description))
            return False
    
    def print_summary(self):
        """Print execution summary"""
        
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}📊 RESUMO DE EXECUÇÃO{RESET}")
        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")
        
        # Stages executed
        print(f"✅ {len(self.stages)} estágios completados com sucesso:")
        for num, desc, status in self.stages:
            print(f"   [{num}] {desc}")
        
        # Failed stages
        if self.failed_stages:
            print(f"\n❌ {len(self.failed_stages)} estágios falharam:")
            for num, desc in self.failed_stages:
                print(f"   [{num}] {desc}")
        
        # Execution time
        print(f"\n⏱️  Tempo total: {minutes}m {seconds}s")
        
        # Results location
        print(f"\n📁 Resultados em:")
        print(f"   - results/metrics/full_factorial_results.csv")
        print(f"   - results/metrics/full_factorial_summary.csv")
        print(f"   - results/figures/*.png")
        print(f"   - models/baseline_model.pkl")
        
        print(f"\n{BOLD}{BLUE}{'='*80}{RESET}\n")
        
        return len(self.failed_stages) == 0

def main():
    """Main orchestration"""
    
    orchestrator = PipelineOrchestrator()
    orchestrator.print_header("DriftSense-PM: Pipeline Completo")
    
    print(f"📝 Cronograma Estimado:")
    print(f"   [1] Feature Engineering (5 min)")
    print(f"   [2] Model Training (2 min)")
    print(f"   [3] Factorial Evaluation (30 min)")
    print(f"   [4] Statistical Analysis (2 min)")
    print(f"   [5] Plot Generation (1 min)")
    
    print(f"💡 Para executar uma etapa isolada:")
    print(f"   python feature_engineering.py")
    print(f"   python train_baseline_full.py")
    print(f"   python master_script.py")
    print(f"   python statistical_analysis.py")
    print(f"   python generate_thesis_plots.py\n")
    
    input(f"{BOLD}Pressione ENTER para iniciar...{RESET}\n")
    
    # Stage 1: Feature Engineering
    success = orchestrator.run_command(
        "python feature_engineering.py",
        "1️⃣ Feature Engineering (Time+Frequency)",
        1
    )
    if not success and len(sys.argv) < 2:
        orchestrator.print_error("Pipeline parou - Feature Engineering falhou")
        return
    
    # Stage 2: Baseline Training
    success = orchestrator.run_command(
        "python train_baseline_full.py",
        "2️⃣ Baseline Model Training (LOF + Evaluation)",
        2
    )
    if not success and len(sys.argv) < 2:
        orchestrator.print_error("Pipeline parou - Model Training falhou")
        return
    
    # Stage 3: Factorial Evaluation
    success = orchestrator.run_command(
        "python master_script.py",
        "3️⃣ Full Factorial Evaluation (54×5=270 configs)",
        3
    )
    if not success and len(sys.argv) < 2:
        orchestrator.print_error("Pipeline parou - Factorial Evaluation falhou")
        return
    
    # Stage 4: Statistical Analysis
    success = orchestrator.run_command(
        "python statistical_analysis.py",
        "4️⃣ Statistical Analysis (Mean±Std, IC, Wilcoxon)",
        4
    )
    if not success and len(sys.argv) < 2:
        orchestrator.print_error("Pipeline parou - Statistical Analysis falhou")
        return
    
    # Stage 5: Plot Generation
    success = orchestrator.run_command(
        "python generate_thesis_plots.py",
        "5️⃣ Generate Publication Plots",
        5
    )
    if not success and len(sys.argv) < 2:
        orchestrator.print_error("Pipeline parou - Plot Generation falhou")
        return
    
    # Print summary
    all_passed = orchestrator.print_summary()
    
    # Exit status
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{RED}❌ Pipeline interrompido pelo utilizador{RESET}")
        sys.exit(1)
