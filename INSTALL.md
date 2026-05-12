# DriftSense-PM: Installation Guide for ACM Artifact Review

**Version:** 1.0  
**Last Updated:** May 7, 2026  
**Author:** Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães  
**Contact:** edp@uminho.pt

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [Quick Start](#quick-start)

---

## System Requirements

### Minimum (Development Environment)

- **OS:** Linux (Ubuntu 20.04+), macOS (10.14+), or Windows 10/11
- **Python:** 3.9+ (3.11 recommended for RPi5)
- **RAM:** 4 GB
- **Disk:** 10 GB (for data + models + results)
- **Internet:** Required for pip package downloads (~200 MB)

### Recommended (Production / Raspberry Pi 5)

- **Hardware:** Raspberry Pi 5 (4GB RAM minimum, 8GB recommended)
- **OS:** Raspberry Pi OS (Bookworm, 64-bit)
- **Storage:** 32 GB microSD card (SSD recommended)
- **Python:** 3.11
- **Additional:** Arduino Pro Smart Industry Kit, USB serial cable

---

## Installation Methods

### Method 1: pip (Fastest)

**Time: ~2-3 minutes**

```bash
# 1. Clone repository
git clone https://github.com/eduardaspereira/DriftSense-PM.git
cd DriftSense-PM

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r env/requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, scipy; print('✅ All packages OK')"
```

**Expected Output:**
```
✅ All packages OK
```

---

### Method 2: Conda (Recommended)

**Time: ~3-5 minutes**

```bash
# 1. Clone repository
git clone https://github.com/eduardaspereira/DriftSense-PM.git
cd DriftSense-PM

# 2. Create conda environment
conda env create -f env/environment.yml

# 3. Activate environment
conda activate driftsense-pm

# 4. Verify
python -c "import pandas, sklearn; print('✅ Environment OK')"
```

**To deactivate:** `conda deactivate`

---

### Method 3: Docker (Cross-Platform)

**Time: ~5-10 minutes** (first build)

```bash
# 1. Build image
docker build -f env/Dockerfile -t driftsense:latest .

# 2. Verify image
docker images | grep driftsense

# 3. Run container (interactive)
docker run -it --rm \
  -v $(pwd)/results:/app/results \
  driftsense:latest \
  /bin/bash

# Inside container:
python scripts/train_baseline_full.py
python scripts/master_script.py
```

---

### Method 4: Raspberry Pi 5 (Edge Deployment)

**Time: ~20-30 minutes**

#### 4.1 System Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11 and build tools
sudo apt-get install -y \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  build-essential \
  git

# Verify Python version
python3.11 --version  # Should be 3.11.x
```

#### 4.2 Project Setup

```bash
# Clone repository
git clone https://github.com/eduardaspereira/DriftSense-PM.git
cd DriftSense-PM

# Create virtual environment with Python 3.11
python3.11 -m venv venv_rpi
source venv_rpi/bin/activate

# Install dependencies (on RPi: slower due to compilation)
pip install --upgrade pip
pip install -r env/requirements.txt

# Note: Installation may take 10-15 min on RPi5 (arm64)
```

#### 4.3 Verify RPi Installation

```bash
python -c "import platform; print(f'Python: {platform.python_version()}, Architecture: {platform.machine()}')"
# Expected: Python: 3.11.x, Architecture: aarch64

# Check data availability
ls data/raw/ | head -3
# Expected: D0_dataset.csv, D1_dataset.csv, ...
```

---

## Verification

### Post-Installation Checks

```bash
# 1. Verify Python packages
python -c "
import pandas as pd
import numpy as np
import sklearn
import scipy
import matplotlib
import seaborn
import yaml
import joblib
print(f'✅ pandas {pd.__version__}')
print(f'✅ numpy {np.__version__}')
print(f'✅ scikit-learn {sklearn.__version__}')
print(f'✅ scipy {scipy.__version__}')
"

# 2. Verify project structure
test -d configs && echo "✅ configs/" || echo "❌ configs/ missing"
test -d data && echo "✅ data/" || echo "❌ data/ missing"
test -d scripts && echo "✅ scripts/" || echo "❌ scripts/ missing"
test -f configs/config.yaml && echo "✅ config.yaml" || echo "❌ config.yaml missing"

# 3. Verify data files
test -f data/raw/D0_dataset.csv && echo "✅ D0_dataset.csv" || echo "⚠️  D0_dataset.csv missing"
test -f data/raw/D1_dataset.csv && echo "✅ D1_dataset.csv" || echo "⚠️  D1_dataset.csv missing"

# 4. Verify models directory (should exist, models to be generated)
test -d models && echo "✅ models/" || mkdir -p models && echo "✅ models/ created"

# 5. Quick sanity check (runs feature engineering)
python scripts/feature_engineering.py && echo "✅ Feature engineering OK" || echo "❌ Feature engineering failed"
```

---

## Troubleshooting

### Issue 1: "No module named 'pandas'"

**Solution:**
```bash
# Verify pip
which pip3  # or pip

# Install missing package
pip install pandas>=1.5.0

# Or reinstall all
pip install -r env/requirements.txt --force-reinstall
```

---

### Issue 2: "Python 3.11 not found" (on RPi)

**Solution:**
```bash
# Check available Python versions
python3 --version
python3.11 --version

# If 3.11 not available:
sudo apt-get install python3.11
python3.11 -m venv venv_new
source venv_new/bin/activate
pip install -r env/requirements.txt
```

---

### Issue 3: Permission Denied (Linux/Mac)

**Solution:**
```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run with explicit python
python scripts/feature_engineering.py  # instead of ./scripts/feature_engineering.py
```

---

### Issue 4: "YAML config file not found"

**Solution:**
```bash
# Verify current directory
pwd  # Should output: .../DriftSense-PM

# Verify config exists
test -f configs/config.yaml && echo "✅ Found" || echo "❌ Not found"

# Check relative paths in script
grep "config.yaml" scripts/master_script.py
# Should show: with open('../configs/config.yaml', 'r')

# If running from wrong directory:
cd /path/to/DriftSense-PM/scripts
python master_script.py  # Run from scripts directory
```

---

### Issue 5: Out of Memory (Raspberry Pi)

**Solution:**
```bash
# Check available RAM
free -h

# Enable swap
sudo dphys-swapfile swapon

# Reduce batch processing (if applicable in code)
# Modify config.yaml: reduce batch_size or window_size
```

---

## Quick Start After Installation

```bash
# 1. Activate environment
source venv/bin/activate  # or: conda activate driftsense-pm

# 2. Run feature extraction (5 min)
cd scripts
python feature_engineering.py

# 3. Train baseline model (2 min)
python train_baseline_full.py

# 4. Execute full factorial with 5 repetitions (30 min on PC, 2-3h on RPi)
python master_script.py --repetitions 5

# 5. Statistical analysis (2 min)
python statistical_analysis.py

# 6. Generate plots (1 min)
python generate_thesis_plots.py

# 7. View results
cat ../results/metrics/full_factorial_summary.csv
ls -lh ../results/figures/
```

---

## Installation Checklist

- [ ] Python 3.9+ installed (`python --version`)
- [ ] Virtual environment created (venv or conda)
- [ ] Dependencies installed (`pip install -r env/requirements.txt`)
- [ ] Config file exists (`test -f configs/config.yaml`)
- [ ] Data files present (`ls data/raw/D*_dataset.csv`)
- [ ] Models directory writable (`test -w models/`)
- [ ] Results directory writable (`test -w results/`)
- [ ] Feature engineering runs without errors
- [ ] Baseline model trains successfully
- [ ] Factorial script produces output file

---

## Next Steps

- **Run Tests:** See [RUN.md](RUN.md) for exact reproduction commands
- **Understand Data:** See [DATASET.md](DATASET.md) for data specification
- **Read Code:** See [REPRODUCIBILIDADE.md](REPRODUCIBILIDADE.md) for architecture overview
- **Generate Paper Plots:** Execute `generate_thesis_plots.py`

---

## Support & Additional Help

- **README.md:** Project overview and quick reference
- **REPRODUCIBILIDADE.md:** Detailed step-by-step reproduction guide (Portuguese)
- **GitHub Issues:** Report bugs at [github.com/eduardaspereira/DriftSense-PM/issues](https://github.com/eduardaspereira/DriftSense-PM/issues)
- **Email:** edp@uminho.pt

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | May 7, 2026 | Initial release for ACM artifact review |
| 0.5 | April 30, 2026 | Candidate version |

---

**Last Modified:** May 7, 2026  
**Maintained By:** Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
