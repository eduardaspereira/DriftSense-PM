# DriftSense-PM: Quick Action Plan

**Objetivo:** Trazer o projeto de Week 13 para Week 15 readiness em 2-3 semanas

---

## 🔴 BLOQUEANTES (Make-or-Break)

### ✅ **Task 1: Correr Fatorial com 5 Repetições**
**Time:** ~3-4 horas (automático, apenas execução)  
**Criticidade:** 🔴 CRÍTICA

```bash
# 1. Editar master_script.py para aceitar parâmetro de repetições
# Adicionar: --repetitions 5

# 2. Correr
cd scripts/
python master_script.py --repetitions 5

# 3. Verificar output
wc -l ../results/metrics/full_factorial_results.csv
# Esperado: 271 linhas (270 dados + header)
```

**Checklist:**
- [ ] `full_factorial_results.csv` tem 270 linhas (não 54)
- [ ] Tem coluna `Repetition` (1-5)
- [ ] Random seeds variáveis por repetição

---

### ✅ **Task 2: Preencher README.md + REPRODUCIBILITY.md**
**Time:** ~1-2 horas  
**Criticidade:** 🔴 CRÍTICA

**README.md (~300 linhas):**
```markdown
# DriftSense-PM

## What is this?
Drift-aware predictive maintenance benchmark for IoT Edge devices.
Tests 5 drift scenarios with 3 detection methods × 3 adaptation strategies.

## Quick Start
```bash
conda env create -f env/environment.yml
conda activate driftsense-pm
python scripts/feature_engineering.py
python scripts/train_baseline_full.py
python scripts/master_script.py --repetitions 5
python scripts/statistical_analysis.py
python scripts/generate_thesis_plots.py
```

## Project Structure
- `data/raw/` - Raw sensor signals (6 scenarios)
- `data/processed/` - Extracted time+frequency features
- `models/` - Trained LOF baseline + scaler
- `scripts/` - Main pipeline
- `results/` - Metrics CSV + publication-ready plots
- `configs/` - Centralized hyperparameters
- `env/` - Dependencies + Docker

## Key Results
- **DET1** (Error Monitoring): ~10 windows detection delay
- **DET2** (KS Test): ~19 windows detection delay
- **A2** (Lightweight): 18× faster than A1 (27ms vs 450ms)

## Metrics Evaluated
- Detection Delay (windows)
- Adaptation Latency (ms)
- Recovery Time (windows)
- False-Positive Rate (%)

## Reproducibility
See `REPRODUCIBILITY.md` for detailed hardware setup, step-by-step validation, and runtime expectations.

## Paper
Submitted to [Conference] as:
> "DriftSense-PM: Efficient Drift Detection and Lightweight Adaptation for Edge Predictive Maintenance"

## Citation
```bibtex
@software{driftsense2026,
  author = {Pereira, E.},
  title = {DriftSense-PM: Drift-Aware Predictive Maintenance Benchmark},
  year = {2026},
  note = {MEI Project, University of Minho}
}
```
```

**Checklist:**
- [ ] README.md > 300 caracteres
- [ ] Tem Quick Start commands
- [ ] Explica cada diretório
- [ ] Cita resultados principais

---

**REPRODUCIBILITY.md (~200 linhas):**
```markdown
# Reproducibility Guide

## Hardware Setup
- **RPi 5** (4GB RAM, USB storage)
- **Arduino Pro Smart Industry Kit** (9x sensors)
- **Nicla Sense ME** (IMU + Environmental)
- **DC Motor + Fan** (vibration source)
- **Serial cable** (/dev/ttyACM0 @ 115200 baud)

## Software Requirements
```bash
# Create environment
conda env create -f env/environment.yml
conda activate driftsense-pm

# Or with pip
pip install -r env/requirements.txt
```

## Step-by-Step Reproduction

### 1. Feature Extraction (5 min)
```bash
python scripts/feature_engineering.py
# Output: 6 CSV files with time+frequency features
# Validation: Each file should have ~1180 rows
ls -lh data/processed/
```

### 2. Baseline Model Training (2 min)
```bash
python scripts/train_baseline_full.py
# Output:
# - models/baseline_model.pkl (LOF with 100 trees)
# - models/scaler.pkl (StandardScaler fitted on D0)
# - results/metrics/report_*.txt (3 detector comparisons)
# - results/figures/cm_*.png (confusion matrices)
```

### 3. Full Factorial Evaluation (30 min for 5 reps)
```bash
cd scripts/
python master_script.py --repetitions 5
# Output: results/metrics/full_factorial_results.csv (270 rows)
cd ..
```

### 4. Statistical Analysis (2 min)
```bash
python scripts/statistical_analysis.py
# Output: results/metrics/full_factorial_summary.csv
# - Mean ± Std for each configuration
# - 95% Confidence Intervals
# - Wilcoxon p-values
```

### 5. Publication Plots (1 min)
```bash
python scripts/generate_thesis_plots.py
# Output:
# - results/figures/fig1_detection_delay.png
# - results/figures/fig2_latency_comparison.png
# - results/figures/fig3_recovery_heatmap.png (if implemented)
```

## Validation Checklist

After running the full pipeline:

```bash
# ✓ Data files
test -f data/raw/D0_dataset.csv && echo "✓ D0 raw"
test -f data/processed/D0_dataset_features.csv && echo "✓ D0 processed"
# ... repeat for D1-D4

# ✓ Models
test -f models/baseline_model.pkl && echo "✓ Model"
test -f models/scaler.pkl && echo "✓ Scaler"

# ✓ Results
test -f results/metrics/full_factorial_results.csv && echo "✓ Factorial"
test -f results/metrics/full_factorial_summary.csv && echo "✓ Summary"

# ✓ Plots
test -f results/figures/fig1_detection_delay.png && echo "✓ Fig1"
test -f results/figures/fig2_latency_comparison.png && echo "✓ Fig2"
```

## Reproducibility Expectations

| Stage | Runtime | Deterministic? | Notes |
|-------|---------|---|------|
| Feature Extraction | 5 min | Yes | Same output every time |
| Baseline Training | 2 min | Yes | Fixed random_state=42 |
| Factorial (1 rep) | 6 min | Yes | Deterministic sklearn |
| Full (5 reps) | 30 min | Yes | Different random seed per rep |
| Analysis | 2 min | Yes | Pure statistics |
| **Total** | **45 min** | **Yes** | **Reproducible ±2%** |

## Troubleshooting

### Issue: `AttributeError: 'StandardScaler' has no attribute 'transform'`
**Solution:** Ensure scaler.pkl was created by `train_baseline_full.py`

### Issue: KS test p-value always high
**Solution:** Check WINDOW_SIZE in config.yaml (should be 20)

### Issue: Detection delay > 50 windows
**Solution:** Adjust DET1 PERSISTENCE or DET2 ALPHA_KS in config.yaml

### Issue: Recovery time = 1.0 (unrealistic)
**Solution:** Verify recovery_threshold in code matches metric definition

## Docker Reproduction

```bash
# Build
docker build -f env/Dockerfile -t driftsense:latest .

# Run full pipeline
docker run --rm driftsense:latest /bin/bash -c \
  "python scripts/feature_engineering.py && \
   python scripts/train_baseline_full.py && \
   python scripts/master_script.py --repetitions 5"

# Extract results
docker run --rm -v $(pwd)/results:/results driftsense:latest
```

## Expected Outputs

### Metrics CSV (full_factorial_results.csv)
```
Scenario,Detector,Adaptation,Delay (Janelas),Latency (ms),Recovery Time
D0,DET0,A0,N/D,0.0,Não Recuperou
D0,DET1,A0,N/D,0.0,Não Recuperou
D1,DET1,A1,9.0,503.2,1.0
...
```
- 270 rows (54 configs × 5 reps)
- All metrics numeric except N/D (not detected)

### Plots (png files)
- fig1_detection_delay.png: Bar chart, Scenario vs Delay, colored by Detector
- fig2_latency_comparison.png: Bar chart, A1 vs A2, with numeric labels
- *(Additional plots if implemented)*

## Performance Baseline

After running on RPi 5:
- Feature extraction: 5-6 min
- Model training: 1-2 min
- Full factorial: 25-35 min (5 reps)

If significantly slower → Check for I/O bottlenecks or model overfitting.

## Artifact Assessment Compatibility

This reproducibility guide ensures alignment with:
- ✓ ACM Artifact Review badges (Functional, Reusable)
- ✓ IEEE Top-Tier Reproducibility Standards
- ✓ Zenodo deposition (post-acceptance)

---

**Status:** Reproducible on any system with Python 3.9+, conda, and the specified dependencies.
```

**Checklist:**
- [ ] REPRODUCIBILITY.md > 200 linhas
- [ ] Tem step-by-step commands
- [ ] Validation checklist
- [ ] Expected runtimes
- [ ] Troubleshooting section

---

## 🟠 HIGH PRIORITY (Next 3 Days)

### ✅ **Task 3: Criar env/requirements.txt**
**Time:** 10 min

```bash
cat > env/requirements.txt << 'EOF'
pandas>=1.5.0,<2.0.0
numpy>=1.23.0,<2.0.0
scikit-learn>=1.2.0,<2.0.0
scipy>=1.9.0,<2.0.0
matplotlib>=3.6.0,<4.0.0
seaborn>=0.12.0,<1.0.0
pyyaml>=6.0,<7.0.0
joblib>=1.2.0,<2.0.0
EOF
```

**Checklist:**
- [ ] requirements.txt existe
- [ ] Versões fixadas (não `==` mas ranges válidos)

---

### ✅ **Task 4: Criar env/environment.yml**
**Time:** 10 min

```bash
cat > env/environment.yml << 'EOF'
name: driftsense-pm
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - pandas>=1.5.0
    - numpy>=1.23.0
    - scikit-learn>=1.2.0
    - scipy>=1.9.0
    - matplotlib>=3.6.0
    - seaborn>=0.12.0
    - pyyaml>=6.0
    - joblib>=1.2.0
EOF
```

**Checklist:**
- [ ] environment.yml existe
- [ ] `conda env create -f env/environment.yml` funciona

---

### ✅ **Task 5: Criar Dockerfile**
**Time:** 15 min

```bash
cat > env/Dockerfile << 'EOF'
FROM python:3.11-slim
LABEL maintainer="Eduardo Pereira <edp@uminho.pt>"
LABEL description="DriftSense-PM: Drift-Aware Predictive Maintenance"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default: run baseline training
CMD ["python", "scripts/train_baseline_full.py"]
EOF
```

**Checklist:**
- [ ] Dockerfile existe
- [ ] `docker build .` sem erros
- [ ] `docker run` executa sem crashes

---

## 🟡 MEDIUM PRIORITY (Next 1 Week)

### ✅ **Task 6: Fix False-Positives em DET2**
**Time:** 1 hora

**Problema:** D0 (cenário sem drift) gera detecção com DET2

**Solução:**
```python
# Em scripts/master_script.py
# Aumentar ALPHA_KS de 0.001 para 0.01
ALPHA_KS = 0.01  # Menos sensível
# Ou aumentar WINDOW_SIZE de 20 para 30

# Re-executar
python scripts/run_all_detectors.py
# Verificar: D0 + DET2 deve ter "Não Detetado" para todas as linhas
```

**Checklist:**
- [ ] D0 + DET2 → 0 detecções (não espúrias)
- [ ] Outros cenários não foram afetados (D1-D4 ainda detectados)

---

### ✅ **Task 7: Criar scripts/statistical_analysis.py**
**Time:** 1-2 horas

```python
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, shapiro
import os

RESULTS_DIR = '../results/metrics/'

# 1. Load raw factorial results
df = pd.read_csv(os.path.join(RESULTS_DIR, 'full_factorial_results.csv'))

# 2. Convert to numeric
df['Delay (Janelas)'] = pd.to_numeric(df['Delay (Janelas)'], errors='coerce')
df['Latency (ms)'] = pd.to_numeric(df['Latency (ms)'], errors='coerce')

# 3. Compute statistics per group
summary = df.groupby(['Scenario', 'Detector', 'Adaptation']).agg({
    'Delay (Janelas)': ['mean', 'std', 'min', 'max', 'count'],
    'Latency (ms)': ['mean', 'std'],
}).round(2)

# 4. Compute 95% CI
def ci_95(group):
    sem = group.sem()  # Standard error of mean
    return 1.96 * sem

summary_ci = df.groupby(['Scenario', 'Detector']).apply(
    lambda x: ci_95(x['Delay (Janelas)'])
)

# 5. Wilcoxon test (DET1 vs DET2)
for scenario in df['Scenario'].unique():
    det1 = df[(df['Scenario'] == scenario) & (df['Detector'] == 'DET1')]['Delay (Janelas)'].dropna()
    det2 = df[(df['Scenario'] == scenario) & (df['Detector'] == 'DET2')]['Delay (Janelas)'].dropna()
    
    if len(det1) > 0 and len(det2) > 0:
        stat, p_val = wilcoxon(det1, det2, alternative='two-sided')
        print(f"{scenario}: Wilcoxon p={p_val:.6f} {'*' if p_val < 0.05 else ''}")

# 6. Save
summary.to_csv(os.path.join(RESULTS_DIR, 'full_factorial_summary.csv'))
print("✅ Statistical summary saved to full_factorial_summary.csv")
```

**Checklist:**
- [ ] Script roda sem erros
- [ ] Output tem Mean ± Std
- [ ] Output tem p-values Wilcoxon

---

### ✅ **Task 8: Criar scripts/run_full_pipeline.py**
**Time:** 30 min

```python
#!/usr/bin/env python3
"""
Complete DriftSense-PM pipeline execution script.
Runs all stages: Feature Extraction → Training → Factorial → Analysis → Plots
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Execute shell command and report status."""
    print(f"\n{'='*60}")
    print(f"📍 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Success: {description}")

def main():
    os.chdir('scripts')
    
    print("🚀 DriftSense-PM Full Pipeline")
    print("=" * 60)
    
    # Stage 1
    run_command("python feature_engineering.py", 
                "1️⃣ Feature Engineering (Time+Frequency)")
    
    # Stage 2
    run_command("python train_baseline_full.py", 
                "2️⃣ Baseline Model Training (LOF + Evaluation)")
    
    # Stage 3
    run_command("python master_script.py --repetitions 5", 
                "3️⃣ Full Factorial Evaluation (54×5=270 configs)")
    
    # Stage 4
    run_command("python statistical_analysis.py", 
                "4️⃣ Statistical Analysis (Mean±Std, IC, Wilcoxon)")
    
    # Stage 5
    run_command("python generate_thesis_plots.py", 
                "5️⃣ Generate Publication Plots")
    
    print("\n" + "="*60)
    print("✨ All stages completed successfully!")
    print("="*60)
    print("📊 Results available in:")
    print("   - results/metrics/full_factorial_results.csv")
    print("   - results/metrics/full_factorial_summary.csv")
    print("   - results/figures/*.png")

if __name__ == "__main__":
    main()
```

**Checklist:**
- [ ] Script roda end-to-end
- [ ] Loga cada stage
- [ ] Validação de outputs

---

## 📋 COMPLETION CHECKLIST

### Week 15 Gate Requirements

- [ ] **Datasets**
  - [ ] D0-D4 raw datasets frozen (DATASET.md atualizado)
  - [ ] Features processed (6 CSV files, cada com ~1180 linhas)
  - [ ] No data modifications after Week 4

- [ ] **Models**
  - [ ] Baseline LOF trained and frozen
  - [ ] StandardScaler persisted
  - [ ] Model performance > 0.8 F1

- [ ] **Experiments**
  - [ ] Full factorial 54 × 5 = 270 configurations executed
  - [ ] All metrics logged (Delay, Latency, FPR, Recovery)
  - [ ] Raw results, not just averages

- [ ] **Statistical Validation**
  - [ ] Mean ± Std computed for all metrics
  - [ ] 95% Confidence Intervals calculated
  - [ ] Wilcoxon signed-rank test DET1 vs DET2
  - [ ] ANOVA for Adaptation strategies

- [ ] **Documentation** (CRÍTICA)
  - [ ] README.md > 500 chars, with Quick Start
  - [ ] REPRODUCIBILITY.md > 200 chars, step-by-step
  - [ ] DATASET.md descreve protocolos de injeção
  - [ ] Inline code comments para componentes complexos

- [ ] **Reproducibility**
  - [ ] requirements.txt com versões fixadas
  - [ ] environment.yml funcional (conda create -f)
  - [ ] Dockerfile buildável (docker build .)
  - [ ] run_full_pipeline.py executa tudo
  - [ ] Git commit tags (v1.0_dataset, v1.0_results)

- [ ] **Quality Assurance**
  - [ ] D0 + DET2 → 0 false-positives
  - [ ] Results within ±5% of previous runs
  - [ ] No hardcoded paths (use config.yaml)
  - [ ] No data files uncommitted to git

- [ ] **Plots (Publication-Ready)**
  - [ ] Detection Delay comparison
  - [ ] Latency A1 vs A2
  - [ ] Recovery time heatmap (opcional)
  - [ ] FPR analysis (opcional)

- [ ] **Artifact Badges (ACM)**
  - [ ] INSTALL.md com setup completo
  - [ ] RUN.md com comandos exatos
  - [ ] Sample dataset (<500 MB) ou traces
  - [ ] Script to regenerate all figures
  - [ ] Hardware setup diagrams

---

## 🎯 Estimated Timeline

| Task | Estimated Time | Start | End |
|------|----------------|-------|-----|
| 1. 5 repetições | 3-4h | Day 1 | Day 1 |
| 2. README + REPRO | 2h | Day 1 | Day 1 |
| 3-5. Deps + Docker | 30m | Day 1 | Day 1 |
| 6. Fix FPR | 1h | Day 2 | Day 2 |
| 7. Statistical script | 1-2h | Day 2 | Day 2 |
| 8. Pipeline script | 30m | Day 2 | Day 2 |
| **Total** | **~10-11h** | **Day 1** | **Day 2** |

**Then:** 2-3 more days for paper draft + artifact assessment

---

**Status:** Ready for execution. Start with Task 1 today.

