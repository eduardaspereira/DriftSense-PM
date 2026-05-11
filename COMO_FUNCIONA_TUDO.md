# 🎓 COMO FUNCIONA TUDO - GUIA COMPLETO DO PROJETO

**Data:** 11 de Maio de 2026  
**Propósito:** Explicar TUDO sobre o DriftSense-PM - para você compreender o que foi feito  
**Audiência:** Você (o responsável do projeto)

---

## 📚 ÍNDICE RÁPIDO

1. [O que é o projeto](#o-que-é-o-projeto)
2. [Estrutura de ficheiros](#estrutura-de-ficheiros)
3. [O que foi feito (cronologia)](#o-que-foi-feito-cronologia)
4. [Como tudo funciona](#como-tudo-funciona)
5. [Como correr o código](#como-correr-o-código)
6. [Como interpretar resultados](#como-interpretar-resultados)

---

## 🎯 O QUE É O PROJETO

**DriftSense-PM** é um **benchmark académico** que testa como algoritmos de detecção funcionam quando os dados mudam (drift).

### Analogia Simples

Imagine um **sensor numa máquina** que mede vibração. No início, funciona bem. Mas depois:
- A temperatura da fábrica aumenta (dados mudam)
- O motor é acelerado (dados mudam)
- O sensor envelhece e começa a dar ruído (dados mudam)

**O projeto testa:** "Se os dados mudam assim, os meus algoritmos de detecção continuam a trabalhar?"

---

## 📂 ESTRUTURA DE FICHEIROS (O QUE TEMOS)

```
DriftSense-PM/
│
├─ 📄 FICHEIROS .MD (DOCUMENTAÇÃO)
│  ├─ README.md                    ← Overview projeto (para público)
│  ├─ INSTALL.md                   ← Como instalar (3 métodos)
│  ├─ RUN.md                       ← Como correr (passo-a-passo)
│  ├─ REPRODUCIBILITY.md           ← Standard ACM (reprodução)
│  ├─ DATASET.md                   ← Explicação dos dados
│  ├─ VALIDACAO_WORKPLAN.md        ← Status vs plano 15 semanas
│  ├─ GUIA_COLEGA_RPi5.md          ← Instruções para colega
│  ├─ RESUMO_EXECUTIVO.md          ← Seu quick reference
│  ├─ RELATORIO_REVISAO_FINAL.md   ← Validação completa
│  └─ workplan.md                  ← Plano original (referência)
│
├─ 🔧 AMBIENTE (COMO INSTALAR)
│  ├─ env/requirements.txt         ← Dependências para pip
│  ├─ env/environment.yml          ← Dependências para conda
│  └─ env/Dockerfile               ← Dependências para docker
│
├─ ⚙️  CONFIGURAÇÃO
│  └─ configs/config.yaml          ← Todos os hiperparâmetros
│
├─ 📊 DADOS (O QUE TESTAMOS)
│  ├─ data/raw/
│  │  ├─ D0_dataset.csv            ← Sem drift (controlo)
│  │  ├─ D1_dataset.csv            ← Com temperatura (drift)
│  │  ├─ D2_dataset.csv            ← Com RPM (drift)
│  │  ├─ D3_dataset.csv            ← Com ruído (drift)
│  │  ├─ D4_D1eD2_dataset.csv      ← Combinado (temperatura + RPM)
│  │  └─ D4_D2eD3_dataset.csv      ← Combinado (RPM + ruído)
│  └─ data/processed/
│     └─ D*_dataset_features.csv   ← Features extraídas (resultado)
│
├─ 🤖 MODELOS (O QUE TREINAMOS)
│  ├─ models/baseline_model.pkl    ← Modelo LOF (aprender anomalias)
│  └─ models/scaler.pkl            ← Normalizador de dados
│
├─ 🐍 SCRIPTS (COMO FUNCIONA O CÓDIGO)
│  ├─ scripts/
│  │  ├─ feature_engineering.py    ← Extrai features dos dados
│  │  ├─ train_baseline_full.py    ← Treina o modelo LOF
│  │  ├─ master_script.py          ← Testa todos os cenários
│  │  ├─ statistical_analysis.py   ← Calcula estatísticas
│  │  ├─ generate_thesis_plots.py  ← Gera gráficos
│  │  ├─ run_full_pipeline.py      ← Corre tudo automaticamente
│  │  ├─ adaptations.py            ← Estratégias de adaptação
│  │  ├─ optimize_detectors.py     ← Otimiza detectores
│  │  ├─ validate_week13_gate.py   ← Valida tudo
│  │  └─ debug/                    ← Scripts debug/troubleshooting
│  │
│  └─ results/
│     ├─ metrics/                  ← Dados dos testes (CSV)
│     │  ├─ full_factorial_results.csv
│     │  ├─ full_factorial_summary.csv
│     │  ├─ wilcoxon_tests.csv
│     │  ├─ adaptation_comparison.csv
│     │  └─ confidence_intervals.csv
│     └─ figures/                  ← Gráficos dos testes (PNG)
│        ├─ fig1_detection_delay.png
│        ├─ fig2_latency_comparison.png
│        └─ (mais 3 gráficos)
│
└─ 📖 paper/
   └─ main.md                      ← Seu paper (em edição)
```

---

## 🔄 O QUE FOI FEITO (CRONOLOGIA)

### FASE 1: RECOLHA DE DADOS (Semanas 1-4)

**O que foi feito:**
- Recolheram 6 datasets (D0-D4) usando um Arduino + sensor
- Cada dataset tem 1180 amostras de vibração

**Ficheiros criados:**
- `data/raw/D0_dataset.csv` até `D4_D2eD3_dataset.csv`

**Resultado:** ✅ Dados prontos para análise

---

### FASE 2: PREPARAÇÃO DE FEATURES (Semana 5)

**O que foi feito:**
- Extraíram 27 features de cada amostra
- Features = informações extraídas dos dados brutos

**Exemplo:**
```
Dado bruto:  [0.5, 0.3, 0.2, 0.1, ...]  (amostra vibração)
    ↓ (feature_engineering.py)
Features:  [media=0.25, std=0.15, max=0.5, ...]  (27 números)
```

**Script:** `scripts/feature_engineering.py`  
**Resultado:** `data/processed/D*_dataset_features.csv`

---

### FASE 3: TREINO DO MODELO (Semana 6)

**O que foi feito:**
- Testaram 3 algoritmos para detectar anomalias
- Selecionaram o melhor (LOF)

**Os 3 testados:**
1. Isolation Forest → F1=0.84 (bom)
2. One-Class SVM → F1=0.79 (ok)
3. **Local Outlier Factor (LOF) → F1=0.91** ✅ MELHOR

**Script:** `scripts/train_baseline_full.py`  
**Resultado:** `models/baseline_model.pkl` (modelo treinado)

---

### FASE 4: TESTES DE DETECÇÃO (Semanas 9-10)

**O que foi feito:**
- Criaram 3 detectores de drift (mudanças nos dados)

**Os 3 detectores:**
1. **DET0** (Baseline) → Não deteta nada (referência)
2. **DET1** (Performance) → Deteta quando F1 score cai abaixo de 0.85
3. **DET2** (Statistical) → Deteta mudança na distribuição dos dados (KS-test)

**Resultado:**
```
DET1: Deteta em 13.5 janelas (rápido!)
DET2: Deteta em 19 janelas (mais lento)
```

---

### FASE 5: ESTRATÉGIAS DE ADAPTAÇÃO (Semanas 11-12)

**O que foi feito:**
- Criaram 3 estratégias para o modelo se adaptar a mudanças

**As 3 estratégias:**
1. **A0** (Nenhuma) → Ignora mudanças (baseline)
2. **A1** (Retraining) → Retreina o modelo a cada 50 janelas (lento: 278ms)
3. **A2** (Lightweight) → Ajuste rápido do modelo (rápido: 10ms) ✅ **27.9× mais rápido!**

**Resultado:**
```
A2 é EXCELENTE para Edge Computing (Raspberry Pi)
```

---

### FASE 6: TESTES COMPLETOS (Semana 13)

**O que foi feito:**
- Testaram TODAS as combinações

**Combinações testadas:**
```
6 datasets × 3 detectores × 3 adaptações = 54 configurações
```

**Script:** `scripts/master_script.py --repetitions 1`  
**Resultado:** `results/metrics/full_factorial_results.csv` (54 linhas)

---

### FASE 7: ANÁLISE ESTATÍSTICA (Semana 14)

**O que foi feito:**
- Analisaram os resultados com testes estatísticos

**Testes realizados:**
```
✅ Wilcoxon signed-rank test    → DET1 vs DET2 são diferentes?
✅ ANOVA                        → A0 vs A1 vs A2 são diferentes?
✅ Confidence Intervals 95%     → Quanto de incerteza existe?
```

**Scripts:**
- `scripts/statistical_analysis.py`
- `scripts/generate_thesis_plots.py`

**Resultado:**
- 4 ficheiros CSV com análises
- 5 gráficos publication-ready

---

## 🔨 COMO TUDO FUNCIONA (TÉCNICO)

### OS 3 CONCEITOS PRINCIPAIS

#### 1. FEATURES (Características extraídas dos dados)

```
Dados brutos (vibração):
  Time: [0.1s, 0.2s, 0.3s, ...]
  Value: [0.5, 0.3, 0.2, ...]

Features extraídas (27 ao total):
  - Mean: 0.33         (valor médio)
  - Std: 0.14          (variabilidade)
  - Max: 0.5           (valor máximo)
  - Min: 0.2           (valor mínimo)
  - RMS: 0.38          (energia)
  - Skewness: 0.12     (assimetria)
  - ... mais 21 features
```

**Importância:** Features = informação comprimida, algoritmos trabalham com isto

#### 2. MODELO LOF (Deteta anomalias)

```
Treino:
  - Vê 100 amostras normais
  - Aprende: "amostras normais são assim"

Predição:
  - Nova amostra chega
  - LOF calcula: "quão diferente é isto?"
  - Se muito diferente → ANOMALIA!
  - Se parecido → NORMAL
```

**Resultado:** F1-score (quanto acerta)

#### 3. DETECTORES DE DRIFT (Detecta mudanças)

```
DET1 (Performance):
  if F1_score < 0.85:
    contador++
  if contador > 10:
    ALERTA: "Drift detectado!"

DET2 (Statistical):
  if KS_test(dados_antigos, dados_novos) < 0.01:
    ALERTA: "Distribuição mudou!"
```

---

## 🚀 COMO CORRER O CÓDIGO

### OPÇÃO 1: TUDO DE UMA VEZ (RECOMENDADO)

```bash
# 1. Instalar dependências
pip install -r env/requirements.txt

# 2. Correr tudo automaticamente
cd scripts
python run_full_pipeline.py

# 3. Ver resultados
cd ../results
ls metrics/
ls figures/
```

**Tempo:** ~45 minutos no PC

---

### OPÇÃO 2: PASSO A PASSO

```bash
cd scripts

# Passo 1: Extrair features (5 min)
python feature_engineering.py

# Passo 2: Treinar modelo (2 min)
python train_baseline_full.py

# Passo 3: Testar todos os cenários (30 min)
python master_script.py --repetitions 1

# Passo 4: Análise estatística (2 min)
python statistical_analysis.py

# Passo 5: Gerar gráficos (1 min)
python generate_thesis_plots.py
```

---

### OPÇÃO 3: UM COMPONENTE POR VEZ

```bash
cd scripts

# Se quer ver features extraídas:
python feature_engineering.py
# Resultado: ../data/processed/D0_dataset_features.csv

# Se quer ver modelo treinado:
python train_baseline_full.py
# Resultado: ../models/baseline_model.pkl

# Se quer só executar 1 repetição:
python master_script.py --repetitions 1
# Resultado: ../results/metrics/full_factorial_results.csv

# Se quer 5 repetições (para RPi5):
python master_script.py --repetitions 5
```

---

## 📊 COMO INTERPRETAR RESULTADOS

### FICHEIRO: `full_factorial_results.csv`

```
Repetition,Scenario,Detector,Adaptation,Delay (Janelas),Latency (ms),Recovery Time
1,D0,DET0,A0,N/D,0.0,Não Recuperou
1,D0,DET1,A0,N/D,0.0,Não Recuperou
1,D0,DET2,A2,19,17.9,1
...
```

**Colunas:**
- `Repetition`: Qual repetição (1-5)
- `Scenario`: Qual dataset (D0-D4)
- `Detector`: Qual detector (DET0/1/2)
- `Adaptation`: Qual adaptação (A0/1/2)
- `Delay`: Quantas janelas até detectar o drift
- `Latency`: Tempo em ms da adaptação
- `Recovery Time`: Se recuperou o desempenho

**O que procurar:**
- DET1 tem delay menor que DET2? → DET1 é melhor
- A2 tem latência menor que A1? → A2 é melhor para Edge

---

### FICHEIRO: `wilcoxon_tests.csv`

```
Scenario,Comparison,p_value,Significant
D1,DET1 vs DET2,0.25,ns
D2,DET1 vs DET2,0.25,ns
...
```

**O que significa:**
- `p_value < 0.05`: São significativamente diferentes
- `p_value > 0.05`: Não são diferentes (ou não há dados suficientes)

---

### FICHEIRO: `adaptation_comparison.csv`

```
Adaptation,Mean_Latency_ms,Std_Latency_ms,Speedup_vs_A1
A0,0.0,0.0,1.0
A1,278.3,14.2,1.0
A2,10.0,9.3,27.9
```

**O que significa:**
- A2 é 27.9× mais rápido que A1
- Para Edge (Raspberry), A2 é ideal

---

### GRÁFICOS (PNGs)

```
fig1_detection_delay.png
  ↓
  Box plot: DET1 vs DET2 (qual é mais rápido?)

fig2_latency_comparison.png
  ↓
  Bar chart: A0 vs A1 vs A2 (qual é mais rápido?)

fig3_recovery_time_heatmap.png
  ↓
  Heatmap: Quanto tempo para recuperar por cenário?

fig4_pareto_front.png
  ↓
  Trade-off: Speed vs Accuracy

fig5_hardware_setup.png
  ↓
  Diagrama: Como está tudo ligado
```

---

## 🔧 BUGS QUE FORAM CORRIGIDOS

### BUG 1: Master Script (11 Maio)

**Problema:** Estavam a faltar 6 configurações (DET0+A2)
- Esperado: 54 configs
- Obtido: 48 configs

**Causa:** Linha 155 tinha `if adapt == 'A2' and det == 'DET0': continue`

**Solução:** Remover essa linha

**Resultado:** ✅ Agora tem 54 configs

---

### BUG 2: UTF-8 Encoding (11 Maio)

**Problema:** Script estatístico crashava no Windows
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680'
```

**Causa:** Windows PowerShell não suporta UTF-8 por default

**Solução:** Adicionar wrapper:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

**Resultado:** ✅ Script funciona no Windows

---

### BUG 3: Path Loop (11 Maio)

**Problema:** Ficheiro não encontrado
```
FileNotFoundError: ../results/metrics/metrics/full_factorial_results.csv
```

**Causa:** Concatenação de paths duplicava "metrics"
```python
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')  # RESULTS_DIR já tinha metrics
filename = os.path.join(METRICS_DIR, 'full_factorial_results.csv')  # duplo!
```

**Solução:** Remover concatenação extra

**Resultado:** ✅ Path correto

---

## ✅ CHECKLIST: TUDO PRONTO?

```
✅ Dados (6 datasets)
✅ Features (27 por amostra)
✅ Modelo (LOF treinado)
✅ Detectores (DET0/1/2)
✅ Adaptações (A0/1/2)
✅ Testes (54 configs × 1 rep testado)
✅ Análise (Wilcoxon, ANOVA, IC95%)
✅ Gráficos (5 plots 300 DPI)
✅ Documentação (9 ficheiros .md)
⏳ 5 repetições completas (colega RPi5)
⏳ Paper finalizado (você)
```

---

## 🎯 PRÓXIMAS FASES

### Responsabilidade COLEGA (Semana 15)

1. Clonar repositório
2. `python master_script.py --repetitions 5`
3. Esperar 2-3 horas
4. Copiar resultados para você

**Tempo:** 3 horas (automatizado)

---

### Responsabilidade VOCÊ (Semana 15)

1. Receber dados de colega
2. Integrar no paper
3. Gerar versão final
4. Submeter

**Tempo:** 2-3 horas

---

## 📞 PERGUNTAS FREQUENTES

### P: "Porque LOF e não outro algoritmo?"

**R:** Testamos 3:
- Isolation Forest: F1=0.84
- One-Class SVM: F1=0.79
- **LOF: F1=0.91** ← Melhor, menos falsos positivos

---

### P: "O que significa 27.9× speedup?"

**R:** A2 é 27.9 vezes mais rápido que A1
```
A1: 278 ms (Retraining completo)
A2: 10 ms (Ajuste rápido)

Speedup = 278 / 10 = 27.9×
```

---

### P: "Porque 54 configurações?"

**R:** Factorial design (testar tudo):
```
6 datasets × 3 detectores × 3 adaptações = 54
```

---

### P: "Como vou saber se resultado está correto?"

**R:** Ver:
1. CSV tem 54 linhas (ou 270 com 5 reps)
2. Colunas têm nomes certos
3. Delay está entre 0-20 janelas
4. Latency está entre 0-300 ms


---

## 🎓 RESUMO FINAL

**O projeto testa:**
- Se algoritmos de detecção funcionam quando dados mudam
- Se estratégias de adaptação conseguem recuperar o desempenho
- Qual é o trade-off entre velocidade e precisão

**O que foi feito:**
1. ✅ Recolheram dados (6 datasets)
2. ✅ Extraíram features (27 características)
3. ✅ Treinaram modelo (LOF F1=0.91)
4. ✅ Criaram detectores (DET1 é 1.4× mais rápido)
5. ✅ Criaram adaptações (A2 é 27.9× mais rápido)
6. ✅ Testaram tudo (54 configs)
7. ✅ Analisaram (Wilcoxon, ANOVA)

**O que falta:**
- ⏳ Colega executar 5 repetições em RPi5
- ⏳ Você integrar no paper

**Status:** 95% Pronto ✅

---

**Este ficheiro foi criado para SUA compreensão. Guarde para referência!**

Data: 11 de Maio de 2026
