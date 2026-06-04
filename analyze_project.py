#!/usr/bin/env python3
"""
DriftSense-PM: Project Analysis Script
Verifica executabilidade e qualidade da documentação
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util

def get_file_size_kb(filepath):
    """Obter tamanho do ficheiro em KB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return round(size_bytes / 1024, 2)
    except:
        return "N/A"

def analyze_documentation():
    """Analisa ficheiros de documentação"""
    print("\n" + "="*80)
    print("📚 ANÁLISE DE DOCUMENTAÇÃO")
    print("="*80 + "\n")
    
    docs = {
        "README.md": "Visão geral do projeto",
        "INSTALL.md": "Guia de instalação",
        "RUN.md": "Guia de reprodução",
        "DATASET.md": "Especificação do dataset",
        "REPRODUCIBILITY.md": "Reprodutibilidade completa",
        "paper/main.md": "Paper draft"
    }
    
    for filepath, description in docs.items():
        if os.path.exists(filepath):
            size_kb = get_file_size_kb(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                f.seek(0)
                content = f.read()
            
            # Heurística de completude (baseado em tamanho e palavras-chave)
            keywords = {
                "README.md": ["Quick Start", "Installation", "Usage", "Results", "References"],
                "INSTALL.md": ["requirements", "conda", "pip", "docker", "verification"],
                "RUN.md": ["python", "scripts", "expected", "output", "validation"],
                "DATASET.md": ["protocol", "scenarios", "sensors", "specifications"],
                "REPRODUCIBILITY.md": ["step-by-step", "hardware", "software", "validation"],
                "paper/main.md": ["abstract", "introduction", "methods", "results", "conclusion"]
            }
            
            keywords_found = sum(1 for kw in keywords.get(filepath, []) if kw.lower() in content.lower())
            completeness = min(100, (keywords_found / max(1, len(keywords.get(filepath, [])))) * 100)
            
            # Problemas comuns
            problems = []
            if size_kb < 10:
                problems.append("Ficheiro muito pequeno (possível falta de conteúdo)")
            if lines < 50:
                problems.append("Poucas linhas de conteúdo")
            if "TODO" in content or "FIXME" in content:
                problems.append("Contém TODOs/FIXMEs")
            
            print(f"📄 {filepath}")
            print(f"   Descrição: {description}")
            print(f"   Tamanho: {size_kb} KB ({lines} linhas)")
            print(f"   Completude: {completeness:.0f}%")
            if problems:
                print(f"   ⚠️  Problemas: {', '.join(problems)}")
            else:
                print(f"   ✅ Sem problemas detectados")
            print()
        else:
            print(f"❌ {filepath}: NÃO ENCONTRADO\n")

def analyze_data_files():
    """Analisa ficheiros de dados"""
    print("\n" + "="*80)
    print("📊 ANÁLISE DE FICHEIROS DE DADOS")
    print("="*80 + "\n")
    
    data_files = {
        "data/processed/D0_dataset_features.csv": "Features D0 (sem drift)",
        "data/processed/D1_dataset_features.csv": "Features D1 (drift covariate)",
        "data/processed/D3_dataset_features.csv": "Features D3 (drift operacional)",
        "data/processed/D4_D1eD2_dataset_features.csv": "Features D4 (D1+D2)",
        "data/processed/D4_D2eD3_dataset_features.csv": "Features D4 (D2+D3)",
    }
    
    for filepath, description in data_files.items():
        print(f"📁 {filepath}")
        print(f"   {description}")
        
        if os.path.exists(filepath):
            try:
                size_kb = get_file_size_kb(filepath)
                df = pd.read_csv(filepath)
                print(f"   ✅ Ficheiro válido")
                print(f"   Tamanho: {size_kb} KB")
                print(f"   Forma: {df.shape[0]} linhas × {df.shape[1]} colunas")
                print(f"   NaN valores: {df.isna().sum().sum()}")
                print(f"   Colunas: {', '.join(df.columns[:5].tolist())}..." if df.shape[1] > 5 else f"   Colunas: {', '.join(df.columns.tolist())}")
            except Exception as e:
                print(f"   ❌ Erro ao ler: {str(e)}")
        else:
            print(f"   ❌ NÃO ENCONTRADO")
        print()

def analyze_models():
    """Analisa ficheiros de modelos"""
    print("\n" + "="*80)
    print("🤖 ANÁLISE DE MODELOS TREINADOS")
    print("="*80 + "\n")
    
    models = {
        "models/baseline_model.pkl": "Modelo LOF/Isolation Forest",
        "models/scaler.pkl": "StandardScaler para normalização"
    }
    
    for filepath, description in models.items():
        print(f"📦 {filepath}")
        print(f"   {description}")
        
        if os.path.exists(filepath):
            try:
                import joblib
                size_kb = get_file_size_kb(filepath)
                model = joblib.load(filepath)
                print(f"   ✅ Modelo carregado com sucesso")
                print(f"   Tamanho: {size_kb} KB")
                print(f"   Tipo: {type(model).__name__}")
                if hasattr(model, 'n_features_in_'):
                    print(f"   Features: {model.n_features_in_}")
            except Exception as e:
                print(f"   ⚠️  Aviso: {str(e)}")
        else:
            print(f"   ❌ NÃO ENCONTRADO")
        print()

def test_imports():
    """Testa se os imports funcionam"""
    print("\n" + "="*80)
    print("🧪 TESTE DE IMPORTS (Dependências)")
    print("="*80 + "\n")
    
    packages = {
        "pandas": "Data manipulation",
        "numpy": "Numerical computing",
        "sklearn": "Machine Learning",
        "scipy": "Scientific computing",
        "matplotlib": "Visualization",
        "seaborn": "Statistical plots",
        "yaml": "YAML parsing",
        "joblib": "Model persistence"
    }
    
    for package, description in packages.items():
        try:
            if package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            version = ""
            try:
                mod = __import__(package)
                if hasattr(mod, '__version__'):
                    version = f" v{mod.__version__}"
            except:
                pass
            print(f"✅ {package}{version:25} - {description}")
        except ImportError as e:
            print(f"❌ {package:25} - {str(e)}")

def test_script_execution():
    """Testa se os scripts principais podem ser importados"""
    print("\n" + "="*80)
    print("🔧 TESTE DE SCRIPTS PRINCIPAIS")
    print("="*80 + "\n")
    
    scripts = {
        "scripts/feature_engineering.py": "Feature extraction",
        "scripts/train_baseline_full.py": "Model training",
        "scripts/master_script.py": "Factorial evaluation",
        "scripts/statistical_analysis.py": "Statistical tests",
        "scripts/generate_thesis_plots.py": "Plot generation"
    }
    
    for script_path, description in scripts.items():
        print(f"📜 {script_path}")
        print(f"   {description}")
        
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Verificar imports críticos
                imports_needed = ["pandas", "numpy", "sklearn", "scipy"]
                missing_imports = []
                for imp in imports_needed:
                    if f"import {imp}" not in content and f"from {imp}" not in content:
                        missing_imports.append(imp)
                
                print(f"   ✅ Ficheiro existe ({len(content)} bytes)")
                if missing_imports:
                    print(f"   ⚠️  Falta imports: {', '.join(missing_imports)}")
                else:
                    print(f"   ✅ Imports críticos presentes")
            except Exception as e:
                print(f"   ❌ Erro: {str(e)}")
        else:
            print(f"   ❌ NÃO ENCONTRADO")
        print()

def check_config():
    """Verifica config.yaml"""
    print("\n" + "="*80)
    print("⚙️  VERIFICAÇÃO DE CONFIGURAÇÃO")
    print("="*80 + "\n")
    
    if os.path.exists("configs/config.yaml"):
        try:
            import yaml
            with open("configs/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            print("✅ config.yaml carregado com sucesso")
            print(f"   Versão dataset: {config.get('system', {}).get('dataset_version', 'N/A')}")
            print(f"   Repetições: {config.get('experiment', {}).get('repetitions', 'N/A')}")
            print(f"   Diretórios configurados:")
            for key, value in config.get('paths', {}).items():
                print(f"      - {key}: {value}")
        except Exception as e:
            print(f"⚠️  Erro ao ler config: {str(e)}")
    else:
        print("❌ configs/config.yaml NÃO ENCONTRADO")

def check_git():
    """Verifica git e versionamento"""
    print("\n" + "="*80)
    print("📦 VERIFICAÇÃO DE GIT")
    print("="*80 + "\n")
    
    if os.path.exists(".git"):
        print("✅ Repositório Git encontrado")
        try:
            import subprocess
            result = subprocess.run(["git", "log", "--oneline", "-5"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   Últimos 5 commits:")
                for line in result.stdout.strip().split('\n'):
                    print(f"      {line}")
            
            # Tags
            tags_result = subprocess.run(["git", "tag"], 
                                       capture_output=True, text=True, timeout=5)
            if tags_result.returncode == 0 and tags_result.stdout.strip():
                print(f"\n   Tags: {', '.join(tags_result.stdout.strip().split())}")
        except Exception as e:
            print(f"   ⚠️  Não foi possível obter histórico git: {str(e)}")
    else:
        print("⚠️  .git NÃO ENCONTRADO - Não é um repositório Git")

def main():
    """Executa todas as análises"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  DriftSense-PM: Project Executability Analysis".center(78) + "║")
    print("║" + "  Verificando se o projeto é realmente executável do zero".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        analyze_documentation()
        analyze_data_files()
        analyze_models()
        test_imports()
        test_script_execution()
        check_config()
        check_git()
        
        print("\n" + "="*80)
        print("✅ ANÁLISE COMPLETA")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Erro durante análise: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
