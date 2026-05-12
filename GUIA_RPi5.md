# 🚀 GUIA PRÁTICO: O QUE A SUA COLEGA PRECISA FAZER NO RASPBERRY PI 5

**Data:** 11 de Maio de 2026  
**Objetivo:** Completar as validações finais de hardware + integrar dados reais no paper  
**Tempo Estimado:** 4-5 horas (principalmente tempo de execução)

---

## 📋 RESUMO EXECUTIVO

Seu colega tem agora **tudo que precisa** para executar o projeto completo em Raspberry Pi 5. O trabalho restante é **principalmente executar código que já está pronto**.

| Tarefa | Tempo | Responsabilidade |
|--------|-------|-----------------|
| Setup em RPi5 | 30 min | Colega |
| Quick test (1 rep) | 30 min | Colega |
| Full factorial (5 reps) | 2-3h | Colega |
| Medição consumo energético | 2-3h | Colega (paralelo) |
| Copiar resultados para PC | 10 min | Colega |
| Integração no paper | 1-2h | Você |
| **TOTAL** | **~7 horas** | **Colega: 3h, Você: 4h** |

---

## ✅ PASSO 1: SETUP INICIAL EM RASPBERRY PI 5 (30 min)

### Pré-requisitos
- [ ] Raspberry Pi 5 com OS instalado (recomendado: Ubuntu 24.04 ou Raspberry Pi OS 64-bit)
- [ ] Acesso via SSH ou terminal local
- [ ] Conexão à internet (para clone + pip install)
- [ ] ~500 MB espaço livre em disco

### Passos

#### 1.1: Clone o repositório
```bash
cd ~
git clone https://github.com/eduardaspereira/DriftSense-PM.git
cd DriftSense-PM
```

#### 1.2: Crie um virtual environment
```bash
python3.11 -m venv venv_rpi
source venv_rpi/bin/activate
```

**Nota:** Se Python 3.11 não está instalado:
```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3.11-dev
```

#### 1.3: Instale dependências
```bash
pip install --upgrade pip
pip install -r env/requirements.txt
```

**Tempo esperado:** ~10-15 minutos (depende da conexão)

#### 1.4: Valide o setup
```bash
python scripts/debug/validate_week13_gate.py
```

**Esperado:**
```
✅ All required data files present
✅ Models loaded successfully
✅ Configuration valid
✅ All paths exist
✅ Pipeline ready to run
```

Se tiver erros, reveja `INSTALL.md` (seção Troubleshooting).

---

## ✅ PASSO 2: QUICK TEST - 1 REPETIÇÃO (30 min)

Antes de fazer a execução completa de 2-3 horas, teste com 1 repetição:

```bash
cd scripts
python master_script.py --repetitions 1
```

**O que acontece:**
- ✅ Executa 54 configurações (6 drift × 3 detector × 3 adaptation)
- ✅ Tempo: ~30-40 minutos em RPi5
- ✅ Output: `../results/metrics/full_factorial_results.csv` com 54 linhas

**Verificação:**
```bash
# Contar linhas (deve ser 54)
wc -l ../results/metrics/full_factorial_results.csv
# Output esperado: 54 full_factorial_results.csv

# Ver primeiras linhas
head -5 ../results/metrics/full_factorial_results.csv
```

**Se tudo OK:** Proceda para PASSO 3  
**Se erro:** Ver debug em `INSTALL.md` → Troubleshooting

---

## ✅ PASSO 3: FULL FACTORIAL RUN - 5 REPETIÇÕES (2-3 horas)

Agora execute a versão completa que vai produzir os resultados finais:

### 3.1: Prepare o ambiente

```bash
cd scripts
mkdir -p ../results/metrics
mkdir -p ../results/figures

# Limpe execuções anteriores (opcional)
# rm ../results/metrics/full_factorial_results.csv
```

### 3.2: Execute com 5 repetições

```bash
# OPÇÃO A: Simples (recomendado)
python master_script.py --repetitions 5

# OPÇÃO B: Com logging detalhado
python master_script.py --repetitions 5 2>&1 | tee execution.log

# OPÇÃO C: Com medição de tempo
time python master_script.py --repetitions 5
```

**Tempo esperado em RPi5:**
- Processor: ~2-3 horas
- CPU usage: ~80-90% (4 cores)
- Memory: ~150-200 MB
- Disk I/O: Baixo (dados em RAM)

**Durante a execução:**
```
✅ Verá output progressivo tipo:
[Stage 1/54] D0 + DET0 + A0 (Rep 1/5)... DONE (125 ms)
[Stage 2/54] D0 + DET0 + A1 (Rep 1/5)... DONE (345 ms)
[Stage 3/54] D0 + DET0 + A2 (Rep 1/5)... DONE (18 ms)
...
[Stage 270/270] D4_D2eD3 + DET2 + A2 (Rep 5/5)... DONE (22 ms)

Total Time: 2h 47m 23s
```

### 3.3: Verifique o resultado

```bash
# Contar linhas (deve ser 270 = 54 configs × 5 reps)
wc -l ../results/metrics/full_factorial_results.csv

# Ver estrutura do CSV
head -1 ../results/metrics/full_factorial_results.csv  # Header
tail -5 ../results/metrics/full_factorial_results.csv  # Últimas linhas

# Estatísticas básicas
tail -20 ../results/metrics/full_factorial_results.csv | cut -d',' -f5-10
```

**Esperado:**
```
scenario,detector,adaptation,repetition,delay_windows,fpr,fnr,latency_ms,f1_recovery
D0,DET0,A0,1,0,0.0,1.0,0,0.45
D0,DET0,A0,2,0,0.0,1.0,0,0.44
...
D4_D2eD3,DET2,A2,5,8,0.02,0.03,22,0.89
```

---

## ⚡ PASSO 4: MEDIÇÃO DE CONSUMO ENERGÉTICO (PARALELO AO PASSO 3)

Enquanto o código está a executar no PASSO 3, execute medições de energia em tempo real.

### 4.1: Hardware - USB Power Meter Setup

**Modelos Comuns Suportados:**
- Sonoff S31 Lite (WiFi) - Recomendado
- Keweisi KWS-MX18 (USB) - Alternativa
- BlitzWolf BW-SHP15 (WiFi)
- Brennenstuhl PM 231E (Universal)

**Ligações:**
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ AC Power    │ ─── │ USB Power    │ ─── │ USB-C Ada.   │ ─── RPi 5
│ (220V)      │     │ Meter        │     │ (5V 5A)      │
└─────────────┘     └──────────────┘     └──────────────┘
                    [MEDIÇÕES AQUI]
```

### 4.2: Monitorização em Paralelo - 3 MÉTODOS

#### Método 1: SOFTWARE DE LOGGING DO POWER METER (Recomendado)

**Para Sonoff S31 Lite / BW-SHP15 (WiFi):**

```bash
# Terminal 1: Execute o código (Passo 3)
cd scripts
python master_script.py --repetitions 5 2>&1 | tee execution_rpi5.log

# Terminal 2: Conecte-se ao power meter e recolha dados
# Utilize app dedicado (Sonoff eWeLink, etc)
# Ou via script Python:
```

**Script Python para Monitorização WiFi:**
```python
# monitoring_wifi_meter.py
import time
import json
import requests
from datetime import datetime

METER_IP = "192.168.1.XXX"  # Altere para IP do seu meter
INTERVAL = 5  # Recolher dados a cada 5 segundos
DURATION = 11400  # 3.17 horas (tempo máximo esperado do passo 3)

data = []
start_time = time.time()

print(f"[{datetime.now()}] Iniciando monitorização de consumo...")
print(f"IP Power Meter: {METER_IP}")
print(f"Intervalo: {INTERVAL}s, Duração: ~{DURATION//3600}h")
print("-" * 70)

try:
    while time.time() - start_time < DURATION:
        try:
            # Exemplar para Sonoff (ajuste conforme seu meter)
            resp = requests.get(f"http://{METER_IP}/api/power", timeout=5)
            
            if resp.status_code == 200:
                power_data = resp.json()
                
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'voltage_v': power_data.get('voltage', 0),
                    'current_a': power_data.get('current', 0),
                    'power_w': power_data.get('power', 0),
                    'energy_kwh': power_data.get('energy', 0)
                }
                
                data.append(record)
                
                # Print em tempo real
                print(f"[{record['timestamp']}] "
                      f"V={record['voltage_v']:.1f}V | "
                      f"I={record['current_a']:.2f}A | "
                      f"P={record['power_w']:.1f}W | "
                      f"E={record['energy_kwh']:.3f}kWh")
        
        except Exception as e:
            print(f"[ERRO] Falha na leitura: {e}")
        
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n[STOP] Monitorização interrompida pelo utilizador")

# Guardar resultados
output_file = f"power_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ {len(data)} medições guardadas em '{output_file}'")

# Estatísticas básicas
if data:
    powers = [d['power_w'] for d in data if d['power_w'] > 0]
    print(f"\nESTATÍSTICAS:")
    print(f"  Potência média: {sum(powers)/len(powers):.1f} W")
    print(f"  Potência máxima: {max(powers):.1f} W")
    print(f"  Potência mínima: {min(powers):.1f} W")
    if data[-1]['energy_kwh'] > 0:
        print(f"  Energia total: {data[-1]['energy_kwh']:.3f} kWh")
```

#### Método 2: USB POWER METER COM ECRÃ DIGITAL

**Para Keweisi KWS-MX18 ou Brennenstuhl (com ecrã):**

```bash
# 1. Conecte o power meter entre AC e RPi5
# 2. Leia o ecrã manualmente a intervalos:
#    - Cada 30 minutos: anote V, A, W, Time de funcionamento
#    - Crie ficheiro manual:

cat > power_log_manual.csv << 'EOF'
timestamp,voltage_v,current_a,power_w,cumulative_time_min
00:00,220.2,0.15,33,0
00:30,220.1,0.82,180,30
01:00,220.0,0.78,172,60
01:30,219.9,0.80,176,90
02:00,220.1,0.15,33,120
02:30,219.8,0.12,26,150
03:00,220.0,0.14,31,180
EOF
```

#### Método 3: SCRIPT MANUAL + VERIFICAÇÃO

```bash
# Se o power meter tiver porta USB/serial:
# 1. Instale pyserial no RPi5:
pip install pyserial

# 2. Crie script de leitura serial:
python << 'EOF'
import serial
import csv
from datetime import datetime

PORT = '/dev/ttyUSB0'  # Altere conforme seu sistema
BAUD = 9600

with serial.Serial(PORT, BAUD, timeout=1) as ser, \
     open('power_meter_log.csv', 'w') as f:
    
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'raw_data'])
    writer.writeheader()
    
    print(f"Lendo dados de {PORT}...")
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'raw_data': line
                })
        except KeyboardInterrupt:
            break
EOF
```

### 4.3: ANÁLISE DOS DADOS RECOLHIDOS

```python
# analyze_power_consumption.py
import json
import pandas as pd
import numpy as np

# Ler dados recolhidos
with open('power_measurements_*.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("=" * 70)
print("ANÁLISE DE CONSUMO ENERGÉTICO - RPi5 DriftSense-PM")
print("=" * 70)

# Estatísticas
print("\n📊 ESTATÍSTICAS GERAIS:")
print(f"  Tempo total: {len(df) * 5 / 3600:.2f} horas")
print(f"  Medições: {len(df)}")
print(f"  Período: {df['timestamp'].min()} → {df['timestamp'].max()}")

# Potência
powers = df['power_w'].dropna()
print(f"\n⚡ POTÊNCIA (Watts):")
print(f"  Média: {powers.mean():.1f} W")
print(f"  Máxima: {powers.max():.1f} W (picos durante retraining A1)")
print(f"  Mínima: {powers.min():.1f} W (idle)")
print(f"  Desvio padrão: {powers.std():.1f} W")

# Energia
if 'energy_kwh' in df.columns:
    energy_total = df['energy_kwh'].iloc[-1]
    print(f"\n🔋 ENERGIA (kWh):")
    print(f"  Total consumido: {energy_total:.3f} kWh")
    print(f"  Custo (0.20€/kWh): {energy_total * 0.20:.2f} €")

# Por Fase
print(f"\n🔄 CONSUMO POR FASE:")
df['phase'] = df['power_w'].apply(lambda x: 
    'Idle' if x < 50 else 
    'Detecção' if x < 150 else 
    'Retraining'
)

for phase in ['Idle', 'Detecção', 'Retraining']:
    subset = df[df['phase'] == phase]
    if len(subset) > 0:
        print(f"  {phase}: {subset['power_w'].mean():.1f} W "
              f"({len(subset)} amostras, {len(subset)*5/60:.1f} min)")

print("\n" + "=" * 70)
```

### 4.4: INTEGRAR DADOS NO PAPER

```bash
# Terminal 1: Execute o script
python master_script.py --repetitions 5

# Terminal 2: Monitorize CPU + temperatura
watch -n 2 'vcgencmd measure_temp; cat /proc/loadavg'
```

### 4.3: Registos esperados

```
⏱️  Tempo total: ~2h 45m
🔌 Consumo médio: ~4-5 W
💾 Pico: ~7-8 W (durante retraining A1)
🌡️  Temperatura CPU: 55-65°C (normal)
```

### 4.4: Guarde os dados

```bash
# Crie um ficheiro de observações
cat > energy_measurements.txt << EOF
Data: $(date)
Duração Total: 2h 45m
Consumo Médio: 4.5 W
Consumo Pico (A1): 7.2 W
Consumo Repouso (A0): 3.1 W
Temperatura Min/Max: 45°C / 62°C
Notas: [suas observações]
EOF
```

---

## 📊 PASSO 5: COPIAR RESULTADOS PARA PC (10 min)

Após a execução completar, copie os dados para análise final:

### 5.1: Via SCP (via SSH)

```bash
# Do seu PC:
scp -r usuario@rpi_ip:~/DriftSense-PM/results ./results_rpi5

# Ou, se na mesma rede local:
scp -r usuario@192.168.1.100:~/DriftSense-PM/results ./results_rpi5
```

### 5.2: Via USB (se preferir)

```bash
# Na RPi5:
cp -r results/ /media/usb/results_rpi5/

# No PC: Cole em: ./results_rpi5/
```

### 5.3: Também copie a tabela de consumo energético

```bash
# Incluir os ficheiros de medições
scp usuario@rpi_ip:~/DriftSense-PM/energy_measurements.txt ./results_rpi5/
scp usuario@rpi_ip:~/DriftSense-PM/scripts/execution.log ./results_rpi5/
```

---

## ✅ PASSO 6: PROCESSAMENTO FINAL (EM SEU PC) - 1-2 horas

Após receber os dados da RPi5, você deve:

### 6.1: Análise Estatística

```bash
# Integre os dados RPi5 com seus dados PC
python scripts/statistical_analysis.py
# Output:
#   - wilcoxon_tests.csv
#   - confidence_intervals.csv
#   - adaptation_comparison.csv
```

### 6.2: Gere plots finais

```bash
# Atualizar plots com todos os dados
python scripts/generate_thesis_plots.py
# Output:
#   - fig1_detection_delay.png
#   - fig2_latency_comparison.png
#   - fig3_recovery_time_heatmap.png
#   - fig4_pareto_front.png
#   - fig5_hardware_setup.png
```

### 6.3: Integração no paper

```bash
# Edite: paper/main.md
# Adicione:
#   1. Imagens das 5 figuras
#   2. Tabelas dos resultados estatísticos
#   3. Latências reais de RPi5
#   4. Consumo energético real
#   5. Discussão dos resultados
```

Exemplo de integração:
```markdown
## Resultados

### Latência de Detecção

| Cenário | PC (ms) | RPi5 (ms) | Speedup |
|---------|---------|----------|---------|
| D0 DET1 | 9.2 | 22.5 | 0.41× |
| D1 DET2 | 18.7 | 45.1 | 0.41× |
| D3 DET1 | 11.3 | 27.8 | 0.41× |

**Observação:** RPi5 é ~2.4× mais lento que PC, conforme esperado.
Energia em repouso: 3.1 W, com A1: 7.2 W.

![Detection Delay Comparison](../results/figures/fig1_detection_delay.png)
```

### 6.4: Gere o paper final

```bash
# Converter markdown para PDF
pandoc paper/main.md -o paper/DriftSense-PM-Final.pdf \
  --pdf-engine=xelatex \
  --include-in-header=paper/preamble.tex

# Ou use um editor markdown (VS Code, Typora, etc)
```

---

## 📦 PASSO 7: CRIAR ARTIFACT PACKAGE PARA ACM

Após tudo estar pronto:

```bash
# Criar diretório limpo
mkdir -p artifact/driftsense-pm-artifact

# Copiar ficheiros essenciais
cd artifact/driftsense-pm-artifact
cp -r ../../scripts ./
cp -r ../../configs ./
cp -r ../../env ./
cp -r ../../data ./
cp -r ../../results ./
cp -r ../../paper ./
cp ../../README.md ./
cp ../../INSTALL.md ./
cp ../../RUN.md ./

# Criar metadata
cat > METADATA.yaml << EOF
name: DriftSense-PM
version: 1.0
authors:
  - Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães (you)
  - [Colega name]
institution: Universidade do Minho, MEI
date: 2026-05-11
status: Ready for publication
description: Drift-aware predictive maintenance benchmark with full factorial evaluation
reproducibility: 5 repetitions, statistical validation, hardware measurements
EOF

# Fazer zip
cd ..
zip -r driftsense-pm-artifact.zip driftsense-pm-artifact/

# Verificar tamanho (deve ser < 50 MB)
ls -lh driftsense-pm-artifact.zip
```

---

## 🎯 CHECKLIST PARA A COLEGA

### Antes de Começar
- [ ] RPi5 tem Python 3.11 instalado
- [ ] Virtual environment criado
- [ ] Dependências instaladas sem erros
- [ ] Validação passou: `validate_week13_gate.py`
- [ ] USB power meter conectado (se tiver)

### Durante a Execução
- [ ] Executou quick test (1 rep) com sucesso
- [ ] Full factorial (5 reps) iniciou
- [ ] Monitorou consumo energético (se aplicável)
- [ ] Não interrompeu a execução (leva 2-3h)
- [ ] Registou temperatura CPU

### Após a Execução
- [ ] full_factorial_results.csv tem 270 linhas
- [ ] Copiou results/ para PC com sucesso
- [ ] Incluiu ficheiro energy_measurements.txt
- [ ] Incluiu logs de execução

### Status Final
- [ ] Entregou dados completos
- [ ] Problema no seu laptop: fecho

---

## 🆘 TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'X'"

```bash
# Reinstale as dependências
pip install -r env/requirements.txt --force-reinstall

# Ou individual:
pip install pandas numpy scikit-learn scipy matplotlib seaborn pyyaml joblib
```

### "ENOSPC: no space left on device"

```bash
# Verificar espaço
df -h

# Limpar cache
pip cache purge
rm -rf ~/.cache

# Se ainda não houver espaço, copie em etapas:
python master_script.py --repetitions 1  # Copia resultado
scp resultado para PC
rm resultado
python master_script.py --repetitions 1  # Repetição 2
...
```

### "Timeout ou script congelado"

```bash
# Verifique se está a correr (em outro terminal)
top  # ou 'htop'

# Se CPU está 0%, verifique logs:
tail -50 execution.log

# Se ainda assim, pode interromper (Ctrl+C) e retomar:
python master_script.py --repetitions 5 --resume-from 3
# (nota: precisa de implementar opção --resume, por enquanto recomeça do 0)
```

### "Latências muito altas em RPi5"

É **esperado** e **normal**:
- PC: ~10-20 ms latência
- RPi5: ~25-50 ms latência
- Razão: Processador mais fraco, thermal throttling

Isto é um **ponto de validação importante** - mostra que o código funciona em hardware limitado.

---

## 📝 ANOTAÇÕES IMPORTANTES

### O que NÃO fazer:
```
❌ Não modifique configs/config.yaml (muda resultados)
❌ Não delete data/ ou models/ durante a execução
❌ Não interrompa durante a execução (pode corromper CSV)
❌ Não use --repetitions > 10 (tempo excessivo)
❌ Não rodar em SSH com conexão instável (pode desconectar)
```

### O que FAZER se desconectar:
```
✅ Reconecte via SSH
✅ Verifique se processo ainda está a correr: ps aux | grep python
✅ Se sim, espere pela conclusão
✅ Se não, pode retomar:
   python master_script.py --repetitions 5
   (continuará do mesmo ponto se usar checkpoint interno)
```

### Otimizações para RPi5:
```
💡 Para acelerar:
   - Use SSD externo em vez de microSD
   - Desative desktop GUI (headless mode)
   - Fixe CPU frequency: echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   
⚠️  Mas não altere config.yaml (afeta reprodutibilidade)
```

---

## 📞 PRÓXIMOS PASSOS APÓS A EXECUÇÃO

1. **Você (no PC):**
   - [ ] Receba dados_rpi5 da colega
   - [ ] Execute statistical_analysis.py
   - [ ] Gere plots finais
   - [ ] Integre dados no paper

2. **Ambos (para o Professor):**
   - [ ] Revisar paper final
   - [ ] Criar artifact package
   - [ ] Submeter: paper + code + results

3. **Possível Follow-up:**
   - [ ] Escrever resumo comparativo PC vs RPi5
   - [ ] Submeter para conferência ACM
   - [ ] Publicação em workshop

---

## ✨ BOAS SORTES!

O seu projeto está **muito bem estruturado**. Tudo que a colega precisa fazer é:

1. **30 min:** Setup
2. **2-3h:** Deixar correr o código
3. **10 min:** Copiar ficheiros

O resto do trabalho (integração, paper, submission) é responsabilidade sua.

Se tiver dúvidas, reveja:
- `INSTALL.md` - Instalação detalhada
- `RUN.md` - Comandos exatos
- `README.md` - Visão geral

---

**Documento Gerado:** 11 de Maio de 2026  
**Tempo de Leitura:** ~15 minutos  
**Próxima Atapa:** Executar em RPi5 ✅
