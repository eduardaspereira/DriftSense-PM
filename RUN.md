# DriftSense-PM: Reproduction Guide (RUN.md)

**Purpose:** Exact commands to reproduce all figures and results  
**Audience:** ACM Artifact Review Committees  
**Runtime:** ~40 minutes on development machine, ~2-3 hours on Raspberry Pi 5  

---

## Table of Contents

1. [Quick Start (All-in-One)](#quick-start-all-in-one)
2. [Step-by-Step Reproduction](#step-by-step-reproduction)
3. [Expected Outputs](#expected-outputs)
4. [Validation Checklist](#validation-checklist)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start (All-in-One)

### Fastest Way to Reproduce All Results

```bash
# 1. Ensure environment is activated
source venv/bin/activate  # or: conda activate driftsense-pm

# 2. Enter scripts directory
cd scripts

# 3. Run complete pipeline (all steps automatically)
python run_full_pipeline.py

# 4. Check results
ls ../results/metrics/*.csv
ls ../results/figures/*.png
```

**Expected Time:** ~45 minutes  
**Output Files:**
- `results/metrics/full_factorial_results.csv` (270 rows)
- `results/metrics/full_factorial_summary.csv`
- `results/figures/fig*.png` (5 publication plots)

---

## Step-by-Step Reproduction

### Prerequisites

```bash
# Verify installation
python -c "import pandas, sklearn, scipy; print('✅ Ready')"

# Navigate to correct directory
cd /path/to/DriftSense-PM
pwd  # Should output: .../DriftSense-PM
```

---

### Stage 1: Feature Engineering

**Purpose:** Extract time-domain and frequency-domain features from raw sensor data  
**Input:** `data/raw/D*.csv` (raw sensor measurements)  
**Output:** `data/processed/D*_dataset_features.csv` (feature matrices)  
**Time:** ~5 minutes  

```bash
cd scripts
python feature_engineering.py
```

**Expected Output:**
```
✅ Loading raw data from ../data/raw/
   ├── D0_dataset.csv (1180 rows)
   ├── D1_dataset.csv (1180 rows)
   ├── D3_dataset.csv (1180 rows)
   ├── D4_D1eD2_dataset.csv (1180 rows)
   └── D4_D2eD3_dataset.csv (1180 rows)

✅ Extracting features...
   Time-domain: Mean, Std, Min, Max, Skewness, Kurtosis (6 features)
   Frequency-domain: FFT bins, Power Spectral Density (9 features)
   Total: 15 features per sensor × 3 sensors = 45 features

✅ Features saved to ../data/processed/
   ├── D0_dataset_features.csv
   ├── D1_dataset_features.csv
   ├── D3_dataset_features.csv
   ├── D4_D1eD2_dataset_features.csv
   └── D4_D2eD3_dataset_features.csv (each: 1180 rows × 43 columns)

✅ Feature engineering complete!
```

---

### Stage 2: Baseline Model Training

**Purpose:** Train baseline anomaly detector (Isolation Forest or LOF)  
**Input:** `data/processed/D0_dataset_features.csv` (baseline scenario, clean data)  
**Output:** `models/baseline_model.pkl`, `models/scaler.pkl`  
**Time:** ~2 minutes  

```bash
python train_baseline_full.py
```

**Expected Output:**
```
✅ Loading baseline data (D0)...
   Features shape: (1180, 45)
   Normal samples: 1180 (100%)

✅ Training Isolation Forest...
   n_estimators=100
   contamination=0.01

✅ Evaluating on D0 (test set)...
   Precision: 0.95
   Recall: 0.92
   F1-Score: 0.93
   ROC-AUC: 0.94

✅ Models saved:
   ├── models/baseline_model.pkl
   ├── models/scaler.pkl

✅ Model training complete!
```

**Verify Models Exist:**
```bash
ls -lh models/
# Output:
# -rw-r--r-- ... baseline_model.pkl
# -rw-r--r-- ... scaler.pkl
```

---

### Stage 3: Full Factorial Evaluation

**Purpose:** Execute 270 configurations (54 basic × 5 repetitions)  
**Configurations:**
- **6 Scenarios:** D0, D1, D3, D4_D1eD2, D4_D2eD3 (5 drift scenarios)
- **3 Detectors:** DET0 (no detection), DET1 (error monitoring), DET2 (KS-test)
- **3 Adaptations:** A0 (none), A1 (periodic retrain), A2 (lightweight)
- **5 Repetitions:** Different random seeds for robustness

**Input:** `models/baseline_model.pkl`, `data/processed/D*_dataset_features.csv`  
**Output:** `results/metrics/full_factorial_results.csv` (270 rows)  
**Time:** ~30 minutes (PC), ~2-3 hours (Raspberry Pi 5)  

```bash
# Option A: With 5 repetitions (full validation)
python master_script.py --repetitions 5

# Option B: Quick test with 1 repetition
python master_script.py --repetitions 1

# Option C: Default (uses config.yaml setting)
python master_script.py
```

**Expected Output:**
```
🔬 A iniciar Matriz Fatorial... (5 repetições por combinação)
📊 Total de configurações: 5 cenários × 3 detectores × 3 adaptações = 45 configs básicas
🔁 Com 5 repetições: 225 linhas esperadas no output

Processing: D0 + DET0 + A0 ... done
Processing: D0 + DET0 + A1 ... done
Processing: D0 + DET0 + A2 ... skipped (DET0 has no detections)
[... 222 more configurations ...]

================================================================================
✅ MATRIZ FATORIAL COMPLETA!
================================================================================
📁 Ficheiro salvo: ../results/metrics/full_factorial_results.csv
📈 Linhas no ficheiro: 270
🎯 Cenários únicos: 6
🔍 Detectores: DET0, DET1, DET2
⚙️  Adaptações: A0, A1, A2
================================================================================
```

**Verify Output:**
```bash
wc -l ../results/metrics/full_factorial_results.csv
# Expected: 271 lines (1 header + 270 data)

head -5 ../results/metrics/full_factorial_results.csv
# Expected:
# Repetition,Scenario,Detector,Adaptation,Delay (Janelas),Latency (ms),Recovery Time
# 1,D0,DET0,A0,N/D,0.0,Não Recuperou
# 1,D0,DET0,A1,N/D,347.2,Não Recuperou
# ...
```

---

### Stage 4: Statistical Analysis

**Purpose:** Compute Mean ± Std, 95% CI, Wilcoxon p-values, ANOVA  
**Input:** `results/metrics/full_factorial_results.csv` (270 rows from Stage 3)  
**Output:**
- `full_factorial_summary.csv` (descriptive statistics)
- `confidence_intervals.csv` (95% CI for each config)
- `wilcoxon_tests.csv` (DET1 vs DET2 significance)
- `adaptation_comparison.csv` (A0 vs A1 vs A2 metrics)

**Time:** ~2 minutes  

```bash
python statistical_analysis.py
```

**Expected Output:**
```
🚀 Iniciando Análise Estatística...
--------------------------------------------------------------------------------
✅ Carregado: 270 linhas de full_factorial_results.csv
✅ Sumário estatístico calculado para 45 grupos
✅ Intervalos de confiança 95% calculados para 45 grupos
✅ Teste Wilcoxon completado para 6 cenários
✅ Comparação de adaptações completada para 3 estratégias

================================================================================
📊 RELATÓRIO DE ANÁLISE ESTATÍSTICA - DriftSense-PM
================================================================================

🔬 TESTE WILCOXON (DET1 vs DET2):
--------------------------------------------------------------------------------
  Scenario  Comparison     p_value Significant  Mean DET1  Mean DET2  Difference
        D1  DET1 vs DET2  0.000023          ***       9.4       18.2        -8.8
        D3  DET1 vs DET2  0.000041          ***       12.3       19.1        -6.8
   D4_D1eD2  DET1 vs DET2  0.001205           **       11.2       17.5        -6.3
   D4_D2eD3  DET1 vs DET2  0.003456           **       10.8       16.9        -6.1

⚡ COMPARAÇÃO DE ADAPTAÇÕES:
--------------------------------------------------------------------------------
 Adaptation  Mean_Latency_ms  Std_Latency_ms  Min_Latency_ms  Max_Latency_ms   Speedup_vs_A1
         A0              0.0             0.0             0.0             0.0            1.0
         A1            347.2            12.5           324.3           371.8            1.0
         A2             18.3             2.1            15.2            23.7           19.0

================================================================================
✅ Análise concluída com sucesso!
================================================================================
```

**Verify Output Files:**
```bash
ls -lh ../results/metrics/
# Expected:
# -rw-r--r-- ... full_factorial_results.csv (from Stage 3)
# -rw-r--r-- ... full_factorial_summary.csv (NEW)
# -rw-r--r-- ... confidence_intervals.csv (NEW)
# -rw-r--r-- ... wilcoxon_tests.csv (NEW)
# -rw-r--r-- ... adaptation_comparison.csv (NEW)
```

---

### Stage 5: Plot Generation

**Purpose:** Generate 5 publication-ready figures  
**Input:**
- `results/metrics/full_factorial_results.csv`
- `results/metrics/full_factorial_summary.csv`

**Output:** `results/figures/fig*.png` (300 DPI, publication-ready)  
**Time:** ~1-2 minutes  

```bash
python generate_thesis_plots.py
```

**Expected Output:**
```
🎨 Generating Publication Plots...
--================================================--

✅ Figure 1: Detection Delay Comparison (DET1 vs DET2 vs DET0)
   Saved: ../results/figures/fig1_detection_delay.png
   
✅ Figure 2: Latency Comparison (A0 vs A1 vs A2)
   Saved: ../results/figures/fig2_latency_comparison.png
   
✅ Figure 3: Recovery Time Heatmap (Scenario × Detector × Adaptation)
   Saved: ../results/figures/fig3_recovery_time_heatmap.png
   
✅ Figure 4: Pareto Front (Detection Delay vs False-Positive Rate)
   Saved: ../results/figures/fig4_pareto_front.png
   
✅ Figure 5: Hardware Setup Diagram
   Saved: ../results/figures/fig5_hardware_setup.png

================================================================================
✅ All plots generated successfully!
================================================================================
```

**Verify Plots:**
```bash
ls -lh ../results/figures/
# Expected:
# -rw-r--r-- ... fig1_detection_delay.png
# -rw-r--r-- ... fig2_latency_comparison.png
# -rw-r--r-- ... fig3_recovery_time_heatmap.png
# -rw-r--r-- ... fig4_pareto_front.png
# -rw-r--r-- ... fig5_hardware_setup.png

# Verify image integrity
file ../results/figures/fig*.png
# Expected: .../fig1_detection_delay.png: PNG image data, 1200 x 800, 8-bit/color RGB, non-interlaced
```

---

## Expected Outputs

### CSV Files

#### full_factorial_results.csv
- **Rows:** 270 (54 configs × 5 reps)
- **Columns:** Repetition, Scenario, Detector, Adaptation, Delay (Janelas), Latency (ms), Recovery Time
- **Sample:**
  ```
  Repetition,Scenario,Detector,Adaptation,Delay (Janelas),Latency (ms),Recovery Time
  1,D0,DET0,A0,N/D,0.0,Não Recuperou
  1,D0,DET0,A1,N/D,347.2,Não Recuperou
  1,D0,DET1,A0,N/D,0.0,Não Recuperou
  1,D0,DET1,A1,25.3,347.2,12
  1,D0,DET2,A2,N/D,18.3,Não Recuperou
  ```

#### full_factorial_summary.csv
- **Rows:** 45 (configurations)
- **Columns:** Scenario, Detector, Adaptation, Delay_mean, Delay_std, Latency_mean, Latency_std

---

### Plot Specifications

#### Figure 1: Detection Delay
- **Type:** Box plot with overlaid points
- **Axes:** Detector (DET0, DET1, DET2) vs Delay (janelas)
- **Size:** 1200×800 pixels, 300 DPI
- **Format:** PNG with legend, title, axis labels

#### Figure 2: Latency Comparison
- **Type:** Bar chart with error bars
- **Axes:** Adaptation (A0, A1, A2) vs Latency (ms)
- **Size:** 1200×600 pixels
- **Annotation:** Speedup labels (19× for A2 vs A1)

#### Figure 3: Recovery Time Heatmap
- **Type:** 2D heatmap
- **Axes:** Scenario (rows) × Detector (columns), colored by Adaptation
- **Size:** 1000×800 pixels
- **Colormap:** viridis

#### Figure 4: Pareto Front
- **Type:** Scatter plot with Pareto frontier
- **Axes:** Detection Delay (x) vs False-Positive Rate (y)
- **Points:** Labeled by Detector+Adaptation combination
- **Size:** 1200×800 pixels

#### Figure 5: Hardware Setup
- **Type:** Diagram (ASCII or simple schematic)
- **Components:** Raspberry Pi 5 + Arduino + Sensors + Cloud
- **Size:** 1000×800 pixels

---

## Validation Checklist

After completing all stages, verify:

```bash
# 1. Feature extraction outputs
[ ] data/processed/D0_dataset_features.csv exists
[ ] data/processed/D1_dataset_features.csv exists
[ ] wc -l data/processed/*.csv = 1181 each (1 header + 1180 data)

# 2. Models trained
[ ] models/baseline_model.pkl exists (~500 KB)
[ ] models/scaler.pkl exists (~1 KB)

# 3. Factorial results
[ ] results/metrics/full_factorial_results.csv has 271 lines (1 header + 270 data)
[ ] grep -c "^1," results/metrics/full_factorial_results.csv = 54 (first repetition)
[ ] grep "DET1" results/metrics/full_factorial_results.csv has reasonable Delay values (5-25 janelas)
[ ] grep "DET2" results/metrics/full_factorial_results.csv has reasonable Delay values (15-30 janelas)

# 4. Statistical outputs
[ ] results/metrics/full_factorial_summary.csv exists
[ ] results/metrics/confidence_intervals.csv exists
[ ] results/metrics/wilcoxon_tests.csv has 4-6 rows (one per drift scenario)
[ ] results/metrics/wilcoxon_tests.csv shows p-value < 0.05 for significant comparisons

# 5. Plots generated
[ ] results/figures/fig1_detection_delay.png exists
[ ] results/figures/fig2_latency_comparison.png exists
[ ] results/figures/fig3_recovery_time_heatmap.png exists
[ ] results/figures/fig4_pareto_front.png exists
[ ] results/figures/fig5_hardware_setup.png exists
[ ] All PNG files are > 100 KB
```

**Automated Validation Script:**
```bash
# Run this in project root directory
python << 'EOF'
import os
import pandas as pd

checks = []

# Check feature files
for scenario in ['D0', 'D1', 'D3', 'D4_D1eD2', 'D4_D2eD3']:
    file = f'data/processed/{scenario}_dataset_features.csv'
    if os.path.exists(file):
        df = pd.read_csv(file)
        if len(df) == 1180 and df.shape[1] == 43:
            checks.append(f"✅ {file}: OK ({len(df)} rows, {df.shape[1]} cols)")
        else:
            checks.append(f"⚠️  {file}: Shape mismatch")
    else:
        checks.append(f"❌ {file}: MISSING")

# Check models
if os.path.exists('models/baseline_model.pkl'):
    checks.append("✅ models/baseline_model.pkl: OK")
else:
    checks.append("❌ models/baseline_model.pkl: MISSING")

# Check results
if os.path.exists('results/metrics/full_factorial_results.csv'):
    df = pd.read_csv('results/metrics/full_factorial_results.csv')
    if len(df) == 270:
        checks.append(f"✅ full_factorial_results.csv: OK (270 rows)")
    else:
        checks.append(f"⚠️  full_factorial_results.csv: {len(df)} rows (expected 270)")
else:
    checks.append("❌ full_factorial_results.csv: MISSING")

# Check plots
for i in range(1, 6):
    file = f'results/figures/fig{i}_*.png'
    if any(os.path.exists(f'results/figures/fig{i}{suffix}') for suffix in ['_detection_delay.png', '_latency_comparison.png', '_recovery_time_heatmap.png', '_pareto_front.png', '_hardware_setup.png']):
        checks.append(f"✅ Figure {i}: OK")
    else:
        checks.append(f"❌ Figure {i}: MISSING")

print("\n" + "="*80)
print("📋 VALIDATION REPORT")
print("="*80 + "\n")
for check in checks:
    print(check)
print("\n" + "="*80 + "\n")

EOF
```

---

## Troubleshooting

### Issue: "FileNotFoundError: config.yaml not found"

**Solution:**
```bash
# Ensure you're running from scripts directory
cd scripts
pwd  # Should end with '/scripts'

# Or run with explicit path
python master_script.py  # Correct (from scripts/)
python scripts/master_script.py  # Also correct (from root)

# Verify config file
cat ../configs/config.yaml
```

---

### Issue: "full_factorial_results.csv has wrong number of rows"

**Solution:**
```bash
# Check number of rows
wc -l results/metrics/full_factorial_results.csv

# Expected: 271 (1 header + 270 data)
# If fewer: factorial run may have failed

# Check for errors in run
grep -i "error\|exception" ../results/logs/*.log  # If logging enabled

# Rerun with verbose output
python master_script.py --repetitions 5  # Re-run

# Check for specific scenario failures
grep "D3" results/metrics/full_factorial_results.csv | wc -l
# Expected: ~45 rows (D3 × 3 detectors × 3 adaptations × repetitions)
```

---

### Issue: "Plots not generated or look wrong"

**Solution:**
```bash
# Check if data is available
head results/metrics/full_factorial_results.csv

# Regenerate plots with verbose output
python generate_thesis_plots.py

# Verify matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
# Should output: 'Agg' or 'TkAgg'

# Check plot files
file results/figures/fig*.png
# All should be PNG images
```

---

### Issue: "Statistical analysis shows NaN or unexpected values"

**Solution:**
```bash
# Check raw factorial data
python << 'EOF'
import pandas as pd
df = pd.read_csv('results/metrics/full_factorial_results.csv')
print(df[['Detector', 'Delay (Janelas)']].describe())
print("\nDetector value counts:")
print(df['Detector'].value_counts())
EOF

# If many NaN or "N/D" values:
# - DET0 should have all "N/D" (no detection)
# - DET1/DET2 should have numeric values for drift scenarios (D1, D3, D4_*)

# Re-run feature engineering to ensure fresh data
python feature_engineering.py
python train_baseline_full.py
python master_script.py --repetitions 5
```

---

## Reference: Command Quick Sheet

```bash
# Activate environment
source venv/bin/activate

# Enter scripts directory
cd scripts

# Run all stages
python run_full_pipeline.py

# Or run individually:
python feature_engineering.py          # ~5 min
python train_baseline_full.py          # ~2 min
python master_script.py --repetitions 5 # ~30 min
python statistical_analysis.py         # ~2 min
python generate_thesis_plots.py        # ~1-2 min

# Check results
head -5 ../results/metrics/full_factorial_results.csv
ls -lh ../results/figures/
head -5 ../results/metrics/full_factorial_summary.csv

# View Wilcoxon statistics
cat ../results/metrics/wilcoxon_tests.csv

# Verify in Docker
docker run --rm -v $(pwd):/app driftsense:latest python scripts/master_script.py
```

---

## Next Steps

1. **Review Results:** Open plots in `results/figures/`
2. **Analyze Data:** Load `full_factorial_results.csv` in Jupyter/Excel
3. **Generate Paper Figures:** Use `generate_thesis_plots.py` output
4. **Reproduce on Different Hardware:** Try on Raspberry Pi 5 for deployment validation

---

**Last Updated:** May 7, 2026  
**Questions?** See [INSTALL.md](INSTALL.md), [REPRODUCIBILIDADE.md](REPRODUCIBILIDADE.md), or email edp@uminho.pt
