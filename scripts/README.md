# Scripts - DriftSense-PM

Documentação dos scripts de processamento, treino e experimentos do projeto **DriftSense-PM**. Este projeto implementa detecção de anomalias e data drift em dados de sensores IoT com estratégias de adaptação em tempo real.

---

## 📋 Índice

1. [Pipeline de Processamento](#pipeline-de-processamento)
2. [Scripts de Treino](#scripts-de-treino)
3. [Scripts de Simulação](#scripts-de-simulação)
4. [Scripts de Análise](#scripts-de-análise)
5. [Scripts de Hardware](#scripts-de-hardware)
6. [Estratégias de Adaptação](#estratégias-de-adaptação)

---

## Pipeline de Processamento

### 1. **feature_engineering.py**
**Objetivo**: Processar dados brutos em features estruturadas

**Entrada**: 
- Arquivos CSV brutos em `../data/raw/`
- Configurações do `../configs/config.yaml`

**Processo**:
- Aplica **sliding windows** com sobreposição configurável
- Extrai features estatísticas: média, desvio padrão, mínimo, máximo, skewness, kurtosis
- Análise espectral: calcula frequência de pico (FFT)
- Normaliza nomes de colunas

**Saída**: 
- Arquivos processados em `../data/processed/*_features.csv`
- Formato: timestamp, cenário, features extraídas por sensor

**Configurações**:
```yaml
feature_engineering:
  window_size: 20        # Tamanho da janela (amostras)
  step_size: 10          # Passo (50% overlap)
sampling_rate_hz: 2.0    # Taxa de amostragem
```

**Execução**:
```bash
python feature_engineering.py
```

---

## Scripts de Treino

### 2. **train_baseline_full.py**
**Objetivo**: Treinar e validar modelos baseline de detecção de anomalias

**Entrada**:
- Dados D0 (cenário normal) em `../data/processed/D0_dataset_features.csv`
- Dados de drift (D1, D2, D3, D4) para validação

**Modelos testados**:
1. **Isolation Forest** (100 estimadores)
2. **Local Outlier Factor (LOF)**
3. **One-Class SVM**

**Processo**:
- Split 80/20 (treino/teste) em dados normais
- Validação cruzada em dados com drift
- Gera matriz de confusão e relatório de classificação
- Salva o modelo vencedor + scaler

**Saída**:
- `../models/baseline_model.pkl` - Modelo serializado
- `../models/scaler.pkl` - StandardScaler para normalização
- `../results/metrics/report_*.txt` - Relatórios por modelo
- `../results/figures/` - Matrizes de confusão visuais

**Execução**:
```bash
python train_baseline_full.py
```

---

## Scripts de Simulação

### 3. **run_all_detectors.py**
**Objetivo**: Executar estratégias de detecção em múltiplos cenários

**Entrada**:
- Modelos e scaler treinados
- Dados processados (D0, D1, D2, D3, D4)
- Configuração de detectores

**Detectores implementados**:
- **DET0**: Cego (sem detecção) - baseline
- **DET1**: Error Monitoring - monitora anomalias consecutivas com persistência
- **DET2**: Distribution Test - testa distribuição de temperatura (KS test) contra baseline D0

**Parâmetros**:
```yaml
detectors:
  det1_error_monitoring:
    persistence: 10        # Número de alarmes consecutivos para disparar
  det2_distribution_test:
    alpha_ks: 0.001        # Threshold p-value para teste KS
  window_size: 20
```

**Saída**:
- Índice de detecção para cada cenário
- Scores de confiança

**Execução**:
```bash
python run_all_detectors.py
```

---

### 4. **optimize_detectors.py**
**Objetivo**: Grid search para otimizar hiperparâmetros dos detectores

**Estratégia**: Testa combinações de:
- `PERSISTENCE_GRID`: [5, 10, 15, 20, 30]
- `ALPHA_KS_GRID`: [0.01, 0.001, 0.0001, 1e-05, 1e-06]

**Entrada**:
- Dados processados (D1, D2, D3, D4) com drift conhecido

**Saída**:
- Relatório com melhor combinação de parâmetros
- Matriz de desempenho (atraso de detecção, taxa de falsos positivos)

**Execução**:
```bash
python optimize_detectors.py
```

---

### 5. **master_script.py** ⭐
**Objetivo**: Orquestrador principal da simulação completa

**Entrada**:
- Modelo treinado e scaler
- Dados processados de cada cenário
- Configurações de detectores e adaptações

**Workflow**:
1. Carrega modelo baseline (LOF) e scaler
2. Para cada combinação (detector × adaptação):
   - Simula fluxo de dados em tempo real
   - Aplica detecção
   - Aplica adaptação se ativada
   - Mede: **detecção**, **latência**, **recuperação**
3. Executa repetições (`config.experiment.repetitions`)
4. Consolida resultados

**Variações testadas**:
- **Detectores**: DET0 (cego), DET1 (error monitoring), DET2 (distribution test)
- **Adaptações**: A0 (nenhuma), A1 (retreino completo), A2 (lightweight)
- **Cenários**: D0 (normal), D1-D4 (drift em temperatura, humidade, ruído)

**Saída**:
- `../results/metrics/drift_results_consolidated.csv` - Resultados finais
- `../results/logs/` - Logs detalhados
- Métricas: atraso, latência, tempo de recuperação

**Métricas principais**:
- **Delay (Janelas)**: Quantas janelas até detecção
- **Latency (ms)**: Tempo de execução do algoritmo
- **Recovery Time**: Tempo até volta à normalidade

**Execução**:
```bash
python master_script.py
```

---

## Scripts de Análise

### 6. **generate_thesis_plots.py**
**Objetivo**: Gerar gráficos de qualidade académica para publicação

**Entrada**:
- `../results/metrics/full_factorial_results.csv` - Resultados consolidados

**Gráficos gerados**:
1. **Detection Delay**: Comparação DET1 vs DET2 por cenário
2. **Latency Cost**: Custo computacional A1 vs A2
3. **Recovery Time**: Tempo de recuperação por adaptação
4. **Adaptation Effectiveness**: Impacto de cada estratégia

**Configuração visual**:
- Estilo académico (seaborn "whitegrid")
- Fontes de publicação
- Resolução 300 DPI
- Paleta de cores consistente

**Saída**:
- `../results/figures/fig1_detection_delay.png`
- `../results/figures/fig2_latency_cost.png`
- Outros gráficos conforme necessário

**Execução**:
```bash
python generate_thesis_plots.py
```

---

## Scripts de Hardware

### 7. **run_experiment.py**
**Objetivo**: Executar experimento em hardware real (Arduino/Portenta)

**Entrada**:
- Conexão serial a microcontrolador
- GPIO pins para LEDs e ventilador
- Configuração do cenário

**Hardware interfaciado**:
- 🟢 LED Verde: Sistema ativado
- 🔴 LED Vermelho: Sistema desativado
- 🌪️ Ventilador: Controle de velocidade (PWM)
- 🔘 Botão: Ativar/Desativar

**Dados colhidos**:
- Temperatura, humidade (DHT22)
- Aceleração 3-eixos (MPU6050)
- Timestamp sincronizado
- ID da amostra

**Configuração**:
```yaml
system:
  serial_port: /dev/ttyACM0
  baud_rate: 115200
  sampling_rate_hz: 2.0
```

**Saída**:
- `../data/raw/dataset_RAW_DATA_v1.0_raw.csv` - Arquivo CSV com dados brutos

**Execução**:
```bash
python run_experiment.py
```

---

## Estratégias de Adaptação

### 8. **adaptations.py**
**Objetivo**: Implementar diferentes estratégias de adaptação online

**Estratégias**:

#### **A0 - No Adaptation (Baseline)**
- Sem qualquer alteração
- Custo energético: NULO
- Latência: 0 ms
- Uso: Baseline para comparação

#### **A1 - Periodic Full Retrain**
- Retreina o modelo a cada `retrain_interval` janelas
- Combina dados históricos (D0) com novo buffer
- Treina 100 árvores (Isolation Forest)
- Custo energético: **ALTO**
- Latência: **ALTA** (centenas de ms)

```yaml
adaptation:
  a1_periodic_retrain:
    retrain_interval: 50   # Retreina a cada 50 janelas
```

#### **A2 - Lightweight Adaptation (Não implementada neste arquivo)**
- Atualização incremental
- Menor custo computacional
- Implementado no master_script.py

**Implementação**:
```python
from adaptations import apply_a0_no_adaptation, apply_a1_periodic_retrain
```

---

## 📊 Fluxo de Execução Recomendado

```
1. feature_engineering.py
   ↓
2. train_baseline_full.py
   ↓
3. optimize_detectors.py (opcional, para tuning)
   ↓
4. master_script.py (simulação completa)
   ↓
5. generate_thesis_plots.py (visualização)
```

---

## 🔧 Configuração Central

Todos os scripts leem `../configs/config.yaml` para parâmetros. Estrutura:

```yaml
paths:
  raw_data_dir: ../data/raw
  processed_dir: ../data/processed
  models_dir: ../models
  results_dir: ../results

feature_engineering:
  window_size: 20
  step_size: 10

system:
  sampling_rate_hz: 2.0
  serial_port: /dev/ttyACM0

detectors:
  det1_error_monitoring:
    persistence: 10
  det2_distribution_test:
    alpha_ks: 0.001

adaptation:
  a1_periodic_retrain:
    retrain_interval: 50

experiment:
  repetitions: 5
```

---

## 📦 Dependências

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
joblib
pyyaml
pyserial (para hardware)
gpiozero (para GPIO)
colorama (para output colorido)
```

**Instalação**:
```bash
pip install -r ../env/requirements.txt
```

---

## 📝 Notas

- ✅ Todos os scripts incluem tratamento de erros
- ✅ Resultados são persistidos em CSV para análise
- ✅ Configuração centralizada em `config.yaml`
- ✅ Compatível com ACM reproducibility badge
- ⚠️ Hardware real requer setup específico (GPIO, serial)
