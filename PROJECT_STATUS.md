# DriftSense-PM: Projeto Status e Análise Detalhada

**Data:** May 7, 2026  
**Projeto:** Drift-Aware Predictive Maintenance Benchmark (MEI, 1st year)  
**Orientador:** Prof. Flávio de Oliveira Silva, Ph.D.

---

## 📋 Sumário Executivo

O projeto **DriftSense-PM** está **~75% completo** no seu estado atual (Week 13 do plano de 15 semanas). A maioria dos componentes críticos foi implementada com qualidade académica aceitável, mas existem **lacunas significativas** em documentação, validação estatística e reprodutibilidade que comprometem a submissão para conferências ACM.

| Aspecto | Status | Progresso |
|--------|--------|-----------|
| **Aquisição de Dados** | ✅ Completo | 6 cenários (D0-D4) recolhidos |
| **Feature Engineering** | ✅ Completo | Pipeline implementado (Time+Freq domains) |
| **Modelo Baseline** | ✅ Completo | LOF treinado e validado |
| **Detecção de Drift** | ✅ Completo | DET0, DET1, DET2 operacionais |
| **Adaptação** | ✅ Completo | A0, A1, A2 implementados |
| **Fatorial Completo** | ⚠️ Parcial | 54 configurações testadas (6 Drift × 3 Detector × 3 Adaptation) |
| **Documentação** | ❌ Crítica | README, REPRODUCIBILITY, resultados detalhados faltam |
| **Validação Estatística** | ⚠️ Parcial | Faltam IC 95%, testes Wilcoxon, análise de significância |
| **Reprodutibilidade ACM** | ⚠️ Parcial | Sem Docker, sem ambiente lock, sem scripts de reprodução |
| **Paper-Ready Plots** | ✅ Completo | 2 figuras principais geradas (fig1, fig2) |

---

## ✅ O QUE FOI FEITO BEM

### 1. **Pipeline de Feature Engineering (Semana 5)**
**Status:** ✅ **Implementação de Alta Qualidade**

```
✓ Extração de features em Time+Frequency domains
✓ Configuração via YAML (reproducível)
✓ Tratamento de edge cases (std=0, dados vazios)
✓ 50% overlap entre janelas (evita leakage)
✓ Logging detalhado com emojis para rastreamento
```

**Arquivo:** [scripts/feature_engineering.py](scripts/feature_engineering.py)

**Métricas extraídas por eixo:**
- Mean, Std, Max, Min, RMS, Skewness, Kurtosis, Peak Frequency

**Qualidade:** Implementação robusta com tratamento de casos extremos.

---

### 2. **Modelo Baseline Pré-treinado (Semana 6)**
**Status:** ✅ **Excelente Qualidade**

```
✓ Local Outlier Factor (LOF) selecionado (melhor F1 entre 3 modelos)
✓ StandardScaler calibrado e persistido
✓ Split cronológico (sem leakage) - 80/20
✓ Avaliação em 3 detectores (IF, OneClass-SVM, LOF)
✓ Matriz de confusão + relatórios detalhados
```

**Arquivo:** [scripts/train_baseline_full.py](scripts/train_baseline_full.py)

**Decisões justificadas:**
- **IF:** Bom AUC, mas instável em dados desbalanceados
- **OneClass-SVM:** Requer tuning intenso, lento
- **LOF (Vencedor):** Melhor equilíbrio entre F1, latência e interpretabilidade

**Artefatos salvos:**
- `models/baseline_model.pkl` (LOF treinado)
- `models/scaler.pkl` (StandardScaler)

---

### 3. **Detecção de Drift - 3 Estratégias (Semanas 9-10)**
**Status:** ✅ **Bem Implementado**

#### **DET0 (Baseline - Sem Deteção)**
- Controlo: não dispara nunca
- Propósito: Medir degradação natural

#### **DET1 (Error Monitoring - Performance-Based)**
```
if F1 < threshold:
    consecutive_alarms++
if consecutive_alarms >= PERSISTENCE:
    DETECT_DRIFT()
```
- **Threshold:** F1 = 0.85
- **Persistence:** 10 janelas (configurável)
- **Vantagem:** Direto ao objetivo (PM)
- **Desvantagem:** Requer labels/proxy

#### **DET2 (Distribution Test - Statistical)**
```
if p_value(KS_TEST) < ALPHA_KS:
    DETECT_DRIFT()
```
- **Teste:** Kolmogorov-Smirnov
- **Alpha:** 0.001 (configurável)
- **Vantagem:** Não requer labels
- **Desvantagem:** Detecção mais tardia

**Arquivos:**
- [scripts/run_all_detectors.py](scripts/run_all_detectors.py) - Simulação básica
- [scripts/master_script.py](scripts/master_script.py) - Integração completa (Det + Adapt)

---

### 4. **Adaptação - 3 Estratégias (Semanas 11-12)**
**Status:** ✅ **Funcional e Comparável**

#### **A0 (Sem Adaptação)**
- **Latência:** 0 ms
- **Custo:** 0 (baseline de degradação)

#### **A1 (Periodic Retraining - Full)**
```
Frequência: cada 50 janelas
Ação: Retrain LOF com D0_histórico + novos dados
Latência: ~450-500 ms
```
- **Pros:** Máxima adaptação
- **Cons:** Custo energético alto (Edge computing inviável)

#### **A2 (Lightweight Adaptation)**
```
Buffer: últimas 20 janelas
Ação: Treina LOF com poucas árvores (contamination=0.01)
Latência: ~27 ms
```
- **Pros:** 18× mais rápido que A1, Edge-friendly
- **Cons:** Menos robusto em drifts extremos

**Arquivo:** [scripts/adaptations.py](scripts/adaptations.py)

**Métricas capturadas:**
- Latência em ms
- Tempo de recuperação (Recovery Time)
- Taxa de falso-positivos

---

### 5. **Cenários de Drift - Taxonomia Completa (Semanas 7-8)**
**Status:** ✅ **Bem Documentado**

| Cenário | Tipo | Como Criado | Ficheiro |
|---------|------|------------|----------|
| **D0** | Controlo | Setup estabilizado 50% RPM | `D0_dataset.csv` |
| **D1** | Covariate | Temperatura ↑ (secador) | `D1_dataset.csv` |
| **D3** | Operational | RPM 50% → 75% | `D3_dataset.csv` |
| **D4** | Sensor Degradation | Ruído Gaussiano injetado | `D4_D1eD2_dataset.csv` |
| **D5** | Combinado | D1 + D3, D1 + D4 | `D4_D2eD3_dataset.csv` |

**Documentação:** [DATASET.md](DATASET.md) - Excelente detalhe técnico

**Dados disponíveis:**
- 6 ficheiros raw (1200 janelas cada)
- 6 ficheiros processados (com features extraídas)

---

### 6. **Configuração YAML Centralizada**
**Status:** ✅ **Profissional e Reproducível**

**Arquivo:** [configs/config.yaml](configs/config.yaml)

```yaml
✓ Paths centralizados (raw, processed, models, results)
✓ Hiperparâmetros de deteção (alpha_ks, persistence)
✓ Hiperparâmetros de adaptação (buffer_size, retrain_interval)
✓ Sistema e versioning (sampling_rate, dataset_version)
```

**Impacto:** Todos os scripts leem desta fonte única → Reproducibilidade garantida

---

### 7. **Resultados Experimentais**
**Status:** ✅ **Dados Brutos Disponíveis**

**Ficheiros de Resultados:**
- `results/metrics/full_factorial_results.csv` - 54 configurações (6D × 3Det × 3Adapt)
- `results/metrics/optimization_results.csv` - Grid search de parâmetros (25 combos)
- `results/metrics/drift_results_consolidated.csv` - Índices de deteção por cenário
- `results/figures/fig1_detection_delay.png` - Gráfico de atraso de deteção
- `results/figures/fig2_latency_comparison.png` - Comparação de latência A1 vs A2

**Dados Capturados:**
- Índice de deteção (em que janela foi detetado?)
- Latência de adaptação (ms)
- Tempo de recuperação (janelas até F1 > 0.85)
- Taxa de falso-positivos (FPR)

---

## ⚠️ QUALIDADE QUESTIONÁVEL

### 1. **Fatorial "Completo" sem Repetições Estatísticas**
**Problema:** O professor exige **5 repetições** por configuração (Week 12 gate), mas o `full_factorial_results.csv` tem apenas **1 repetição**

```
Esperado: 54 configs × 5 reps = 270 linhas
Atual:    54 configs × 1 rep  = 54 linhas
```

**Impacto:** 
- ❌ Impossível calcular intervalos de confiança (IC 95%)
- ❌ Teste de significância Wilcoxon não aplicável
- ❌ Viola gatekeep Week 12 ("Minimum 5 repetitions successfully logged")

---

### 2. **Detecção Anómala em D0 (Falso-Positivos Não Nulos)**
**Problema:** No cenário D0 (sem drift), DET2 gera **detecção espúria**:

```
D0,DET2,A0,19.0,0.0,Não Recuperou ← Isto é um FP!
```

Esperado: `N/D` (not detected) em D0 com todos os detectores.

**Causa provável:** 
- ALPHA_KS=0.001 muito apertado
- Flutuação natural no baseline (1200 janelas) ultrapassa limiar
- Teste KS esperava distribuições idênticas

**Impacto:**
- ⚠️ FPR não zero → Validação incompleta
- ⚠️ DET2 precisa recalibração

---

### 3. **Recovery Time Sempre 1.0 ou "Não Recuperou"**
**Problema:** Padrão suspeito nos dados:

```
DET1,A2: Recovery Time = 1.0 (todas as linhas)
DET0,A0: Recovery Time = "Não Recuperou" (todas as linhas)
```

**Questões:**
- O `Recovery Time` está **realmente** sendo medido ou é valor hardcoded?
- O critério de recuperação (F1 > 0.85) é atingido tão rápido assim?

**Verificação necessária:** Código em `master_script.py` que calcula `recovery_time`

---

### 4. **Scripts "Master" com Lógica Acoplada**
**Problema:** 
- `master_script.py` tem 150+ linhas com lógica complexa
- Mix de Det + Adapt + Logging tudo junto
- Difícil de debugar e reutilizar

**Exemplo de acoplamento:**
```python
# Dentro de simulate_stream():
if detection_idx is None:
    # ... lógica de deteção ...
    
if adapted_once:
    # ... lógica de adaptação ...
    
# ... métricas de recuperação ...
```

**Impacto:**
- ⚠️ Código "spaghetti" - não modular
- ⚠️ Difícil testar componentes isoladamente
- ⚠️ Violação do princípio SRP (Single Responsibility)

---

## ❌ O QUE FALTA FAZER

### **Crítico para Submissão ACM (Week 15)**

### 1. **Repetições Estatísticas (BLOQUEANTE)**
**Prioridade:** 🔴 **CRÍTICA**

**O que falta:**
```
5 repetições por configuração (54 × 5 = 270 execuções)
+ Random seed variável por repetição
+ Raw metrics armazenados linha-a-linha
+ Cálculo de Mean ± Std
+ IC 95% e testes Wilcoxon
```

**Ficheiro Afetado:** `results/metrics/full_factorial_results.csv`

**Esforço:** 2-3 horas (automático, mas computacionalmente intensivo)

**O que fazer:**
```python
for repetition in range(1, 6):
    for scenario in SCENARIOS:
        for detector in DETECTORS:
            for adaptation in ADAPTATIONS:
                results.append({
                    'Repetition': repetition,
                    'Scenario': scenario,
                    'Detector': detector,
                    'Adaptation': adaptation,
                    'Delay': compute_delay(),
                    'Latency': compute_latency(),
                    'FPR': compute_fpr(),
                    'Recovery': compute_recovery()
                })
results_df.to_csv('full_factorial_results.csv')
```

---

### 2. **Documentação README + REPRODUCIBILITY (CRÍTICO)**
**Prioridade:** 🔴 **CRÍTICA**

**Ficheiros vazios:**
- ❌ `README.md` (0 bytes)
- ❌ `REPRODUCIBILITY.md` (0 bytes)

**O que deve conter:**

#### **README.md**
```markdown
# DriftSense-PM: Drift-Aware Predictive Maintenance Benchmark

## Quick Start
1. Clone repositório
2. conda env create -f env/environment.yml
3. python scripts/train_baseline_full.py
4. python scripts/master_script.py
5. python scripts/generate_thesis_plots.py

## Project Overview
- 15-week MEI project on IoT Edge Computing
- Dataset: Arduino Pro Smart Industry Kit
- Tested 5 drift scenarios across 3 detectors × 3 adaptations

## Structure
data/raw/ → data/processed/ → models/ → results/

## Papers & References
[Citar literatura sobre concept drift, edge adaptation, etc.]
```

#### **REPRODUCIBILITY.md**
```markdown
# Reprodução Completa

## Hardware Setup
- Raspberry Pi 5 (Training/Detection)
- Arduino Pro Smart Industry Kit (Sensors)
- Serial: /dev/ttyACM0 @ 115200 baud
- Sampling: 2 Hz

## Step-by-Step
1. Raw Data Collection: python scripts/run_experiment.py --scenario D0
2. Feature Extraction: python scripts/feature_engineering.py
3. Baseline Training: python scripts/train_baseline_full.py
4. Drift Simulation: python scripts/master_script.py --reps 5
5. Statistical Analysis: [script a criar]
6. Plot Generation: python scripts/generate_thesis_plots.py

## Expected Runtime
- Feature engineering: ~5 min
- Baseline training: ~2 min
- Full factorial (5 reps): ~30 min
- Total: ~45 min

## Validation Checklist
□ Full factorial results have 270 rows (54 configs × 5 reps)
□ Mean ± Std computed
□ Wilcoxon p-values < 0.05 for significant diffs
□ All plots regenerated with identical outputs
```

**Esforço:** 1-2 horas

---

### 3. **Dockerfile + requirements.txt + environment.yml**
**Prioridade:** 🟠 **ALTA**

**Atualmente faltam:**
- ❌ `env/requirements.txt` vazio
- ❌ `env/environment.yml` não existe
- ❌ `Dockerfile` não existe

**O que criar:**

**`env/requirements.txt`:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
joblib>=1.2.0
```

**`env/environment.yml`:**
```yaml
name: driftsense-pm
channels:
  - conda-forge
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
```

**`Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY env/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "scripts/train_baseline_full.py"]
```

**Esforço:** 30 min

---

### 4. **Script de Análise Estatística Completo**
**Prioridade:** 🟠 **ALTA**

**Ficheiro:** Criar `scripts/statistical_analysis.py`

**O que deve fazer:**
```python
✓ Carregar full_factorial_results.csv
✓ Computar Mean ± Std para cada config
✓ Teste Wilcoxon (DET1 vs DET2)
✓ ANOVA para Adaptation strategies
✓ Tabelas LaTeX para paper
✓ Salvar CSV com IC 95%
```

**Exemplo output:**
```
Detector | Scenario | Delay (Mean) | Delay (Std) | IC_95_Lower | IC_95_Upper | p_value_vs_DET2
DET1     | D1       | 9.2          | 0.45        | 8.8         | 9.6         | 0.023*
DET2     | D1       | 19.0         | 0.10        | 18.9        | 19.1        | -
```

**Esforço:** 1-2 horas

---

### 5. **Scripts de Reprodução Validados**
**Prioridade:** 🟠 **ALTA**

**Criar `scripts/run_full_pipeline.sh` ou `.py`:**
```bash
#!/bin/bash
set -e

echo "1️⃣ Feature Engineering..."
python scripts/feature_engineering.py

echo "2️⃣ Baseline Training..."
python scripts/train_baseline_full.py

echo "3️⃣ Full Factorial (5 reps)..."
python scripts/master_script.py --repetitions 5

echo "4️⃣ Statistical Analysis..."
python scripts/statistical_analysis.py

echo "5️⃣ Generate Plots..."
python scripts/generate_thesis_plots.py

echo "✅ Pipeline concluído!"
```

**Esforço:** 30 min

---

### 6. **Verificação de False-Positives (DET2 em D0)**
**Prioridade:** 🟡 **MÉDIA**

**Problema:** DET2 deteta drift em D0 (cenário sem drift)

**Ação:**
1. Analisar distribuição de `Temp_Mean` em D0
2. Ajustar ALPHA_KS ou WINDOW_SIZE
3. Re-executar fatorial com parâmetros corrigidos

**Verificação:**
```python
df = pd.read_csv('results/metrics/full_factorial_results.csv')
fps = df[(df['Scenario'] == 'D0') & (df['Detector'] == 'DET2') & (df['Detection_Index'] != 'Não Detetado')]
print(f"False positives em D0 com DET2: {len(fps)}")
# Esperado: 0
```

**Esforço:** 1 hora

---

### 7. **Paper Draft + Figures de Publicação**
**Prioridade:** 🟡 **MÉDIA**

**Criar `paper/main.md` ou `paper/main.tex`:**

**Secções obrigatórias:**
1. **Abstract** (200 palavras)
2. **Introduction** - Motivação de Edge + Drift + PM
3. **Related Work** - Concept Drift, Anomaly Detection, PM systems
4. **Methods** - Taxonomia de drifts, detectores, adaptações
5. **Experimental Design** - Full factorial, repetições, métricas
6. **Results** - Tabelas, gráficos, análise estatística
7. **Discussion** - Trade-offs, limitações, insights
8. **Conclusion + Future Work**

**Figuras críticas a adicionar:**
- ✅ Fig1: Detection Delay (já existe)
- ✅ Fig2: Latency Comparison (já existe)
- ❌ Fig3: Recovery Time heatmap (Scenario × Detector × Adaptation)
- ❌ Fig4: FPR vs Detection Delay (Pareto front)
- ❌ Fig5: Hardware setup (foto + diagrama)
- ❌ Fig6: Conceptual drift taxonomy

**Esforço:** 4-6 horas

---

### 8. **Artifact Evaluation Checklist (ACM)**
**Prioridade:** 🟡 **MÉDIA**

**Checklist para submissão (criar `ARTIFACT.md`):**

```markdown
# Artifact Evaluation Checklist

## ✓ COMPLETENESS
- [ ] Source code available
- [ ] All configurations documented
- [ ] Sample dataset provided (<500 MB)
- [ ] Scripts to regenerate all plots
- [ ] Raw metrics for all configurations

## ✓ CONSISTENCY
- [ ] Results reproducible within ±5%
- [ ] All runs use fixed random seeds
- [ ] No data modifications between runs
- [ ] Version control with git

## ✓ DOCUMENTATION
- [ ] INSTALL.md (setup instructions)
- [ ] RUN.md (exact commands for figures)
- [ ] README.md (project overview)
- [ ] Inline code comments

## ✓ FUNCTIONALITY
- [ ] Dockerfile builds without errors
- [ ] Pipeline runs end-to-end
- [ ] All plots generate successfully
- [ ] Statistical tests execute

## ✓ REUSABILITY
- [ ] Modular code structure
- [ ] Hyperparameter sensitivity analysis
- [ ] Generalizability discussed
- [ ] Extension points identified
```

**Esforço:** 1-2 horas

---

## 🎯 PLANO DE AÇÃO PARA COMPLETAR

### **Timeline Estimada: 2-3 Semanas**

| Prioridade | Task | Esforço | Dependências |
|-----------|------|--------|-------------|
| 🔴 CRÍTICA | 1. Repetições estatísticas (5 reps) | 3h | ✓ Scripts prontos |
| 🔴 CRÍTICA | 2. README + REPRODUCIBILITY | 2h | ✓ Documentos prontos |
| 🟠 ALTA | 3. Docker + Deps | 1h | 2 |
| 🟠 ALTA | 4. Script de análise estatística | 2h | 1 |
| 🟠 ALTA | 5. Script de reprodução completo | 1h | 1, 2 |
| 🟡 MÉDIA | 6. Verificar FPR (D0/DET2) | 1h | 1 |
| 🟡 MÉDIA | 7. Paper draft | 5h | 1, 4 |
| 🟡 MÉDIA | 8. Artifact checklist | 2h | 7 |

**Total:** ~17-18 horas de trabalho

---

## 📊 TABELA RESUMIDA: O QUE FOI FEITO vs. O QUE FALTA

| Componente | Weeks | Status | Ficheiros | Qualidade | Falta |
|-----------|-------|--------|-----------|-----------|-------|
| **Data Collection** | 1-8 | ✅ Completo | 6 raw CSVs | ⭐⭐⭐⭐ | - |
| **Feature Engineering** | 5 | ✅ Completo | `feature_engineering.py` | ⭐⭐⭐⭐ | - |
| **Baseline Model** | 6 | ✅ Completo | `train_baseline_full.py` | ⭐⭐⭐⭐ | - |
| **Drift Detection** | 9-10 | ✅ Completo | `run_all_detectors.py` | ⭐⭐⭐ | FP fix |
| **Adaptation** | 11-12 | ✅ Completo | `adaptations.py` | ⭐⭐⭐⭐ | - |
| **Factorial Exp** | 13 | ⚠️ Parcial | `master_script.py` | ⭐⭐⭐ | 5 reps |
| **Statistical Anal** | 14 | ❌ Missing | - | - | ✅ Criar |
| **Plots** | 14 | ✅ Parcial | `generate_thesis_plots.py` | ⭐⭐⭐ | 4+ figs |
| **Documentation** | 15 | ❌ Crítica | - | 0% | ✅ Crítico |
| **Dockerfile** | 15 | ❌ Missing | - | - | ✅ Criar |
| **Reproducibility** | 15 | ⚠️ Parcial | Alguns scripts | ⭐⭐ | ✅ Muito |
| **Paper Draft** | 15 | ❌ Missing | - | - | ✅ Criar |

---

## 🚀 COMO PROCEDER

### **Fase 1: Corrigir Critério de Aceitação (Week 15 Gate)**

**Comando para rodar 5 repetições:**
```bash
cd scripts/
python master_script.py --repetitions 5 --output ../results/metrics/full_factorial_v1.csv
```

✅ **Resultado esperado:** 
- `full_factorial_v1.csv` com 270 linhas (54 × 5)
- Coluna adicional: `Repetition` (1-5)

---

### **Fase 2: Documentação**

**1. Preencher `README.md`:**
```bash
Copiar template de ../templates/README_template.md
Adaptar para DriftSense-PM
```

**2. Preencher `REPRODUCIBILITY.md`:**
```bash
Adicionar:
- Hardware setup diagrama
- Serial port config
- Step-by-step validation
- Runtime expectations
```

---

### **Fase 3: Análise Estatística**

**Criar `scripts/statistical_analysis.py`:**
```python
import pandas as pd
from scipy.stats import wilcoxon, describe
import numpy as np

# Carregar dados
df = pd.read_csv('../results/metrics/full_factorial_v1.csv')

# Computar Mean ± Std
summary = df.groupby(['Scenario', 'Detector', 'Adaptation']).agg({
    'Delay (Janelas)': ['mean', 'std', 'count'],
    'Latency (ms)': ['mean', 'std'],
    'Recovery Time': ['mean', 'std']
}).round(2)

# IC 95%
summary['Delay_IC95'] = 1.96 * summary['Delay (Janelas)']['std'] / np.sqrt(5)

# Teste Wilcoxon
det1_data = df[df['Detector'] == 'DET1']['Delay (Janelas)']
det2_data = df[df['Detector'] == 'DET2']['Delay (Janelas)']
stat, p_val = wilcoxon(det1_data, det2_data)

print(summary)
print(f"Wilcoxon p-value (DET1 vs DET2): {p_val:.6f}")
```

---

### **Fase 4: Validação Final**

**Checklist antes de enviar:**

- [ ] Full factorial tem 270 linhas (54 configs × 5 reps)
- [ ] `README.md` > 500 chars
- [ ] `REPRODUCIBILITY.md` com step-by-step
- [ ] `Dockerfile` buildável (`docker build .`)
- [ ] `requirements.txt` completo (pip freeze)
- [ ] Todas as plots regeneráveis com 1 comando
- [ ] Código comentado (especialmente adaptations.py e master_script.py)
- [ ] Sem ficheiros de dados temporários (gitignore atualizado)
- [ ] All metrics stored as CSV (not pictures)

---

## 📝 NOTAS FINAIS

### **Pontos Fortes do Projeto**
1. ✅ Taxonomia de drifts bem definida e executada
2. ✅ Implementação clara de detectores e adaptações
3. ✅ Features extraídas corretamente (Time+Freq)
4. ✅ Configuração centralizada (YAML)
5. ✅ Dados brutos e processados preservados

### **Pontos Críticos para Melhoria**
1. ❌ **Faltam repetições estatísticas** → Impossível submeter sem isto
2. ❌ **Documentação vazia** → ACM rejeita automaticamente
3. ❌ **False-positives em D0** → Validação incompleta
4. ❌ **Código acoplado** → Difícil debugar e reutilizar
5. ❌ **Sem Dockerfile** → Não reproducível fora do RPi

### **Recomendações Finais**
- ✅ Priorizar **5 repetições** (faz a diferença em IC e testes)
- ✅ Refatorar `master_script.py` em funções modularizadas
- ✅ Criar teste unitário para cada detector/adaptação
- ✅ Usar Git tags para versão de dataset (`v1.0`, etc.)
- ✅ Preparar artifact.zip para Zenodo (pós-aceitação)

---

**Status Final:** O projeto é **academicamente válido** mas precisa de **2-3 semanas** de trabalho de "finishing touches" para ACM badge readiness.

**Estimativa de Sucesso (com as correções):** 85-90% (dependendo do rigor dos revisores)

---

*Document generated: May 7, 2026*  
*Project: DriftSense-PM (MEI, Week 13/15)*  
*Analyzer: GitHub Copilot*

