# 📋 CHECKLIST: O QUE FALTA NO PROJETO DRIFTSENSE-PM

**Data:** 7 Maio 2026  
**Semana:** 13/15 (75% completo)  
**Contexto:** Merge conflicts resolvidos, 4 docs análise integrados, Project congelado Week 4  

---

## 🔴 CRÍTICO (Blocking Week 15 Gate) - ~8-10 horas

### 1. **Executar Fatorial com 5 Repetições** ⏳
**Status:** ❌ FALTA  
**Impacto:** BLOQUEANTE - impossível Week 15 gate sem isto  
**Localização:** `scripts/master_script.py`  
**O que falta:**
- [ ] Modificar `master_script.py` para aceitar parâmetro `--repetitions 5`
- [ ] Loop de 5 repetições com random seed variável por rep
- [ ] Coluna adicional "Repetition" (1-5) no CSV
- [ ] Arquivo output: `results/metrics/full_factorial_results.csv`

**Status Esperado:**
- Atualmente: 54 linhas (54 configs × 1 rep)
- Esperado: 270 linhas (54 configs × 5 reps)

**Tempo Estimado:** 3-4 horas (execução automática, apenas setup)

---

### 2. **Criar env/requirements.txt** ⏳
**Status:** ❌ FALTA (ficheiro vazio)  
**Impacto:** CRÍTICO - impossível Docker/reproducibilidade  
**Localização:** `env/requirements.txt`  
**O que falta:**
- [ ] Listar todas dependências Python com versões fixadas
- [ ] pandas>=1.5.0,<2.0.0
- [ ] numpy>=1.23.0,<2.0.0
- [ ] scikit-learn>=1.2.0,<2.0.0
- [ ] scipy>=1.9.0,<2.0.0
- [ ] matplotlib>=3.6.0,<4.0.0
- [ ] seaborn>=0.12.0,<1.0.0
- [ ] pyyaml>=6.0,<7.0.0
- [ ] joblib>=1.2.0,<2.0.0

**Tempo Estimado:** 10 minutos

---

### 3. **Criar env/environment.yml** ⏳
**Status:** ❌ FALTA (não existe)  
**Impacto:** CRÍTICO - conda reproducibilidade  
**Localização:** `env/environment.yml`  
**O que falta:**
- [ ] Ficheiro conda environment completo
- [ ] Python 3.11 pinned
- [ ] Todas dependências via pip em "dependencies"
- [ ] Nome: driftsense-pm

**Tempo Estimado:** 10 minutos

---

### 4. **Criar env/Dockerfile** ⏳
**Status:** ❌ FALTA (não existe)  
**Impacto:** CRÍTICO - Docker reproducibilidade  
**Localização:** `env/Dockerfile`  
**O que falta:**
- [ ] Base image: python:3.11-slim
- [ ] COPY env/requirements.txt .
- [ ] RUN pip install -r requirements.txt
- [ ] COPY . .
- [ ] CMD para rodar pipeline

**Tempo Estimado:** 15 minutos

---

### 5. **Preencher README.md com Conteúdo Completo** ⏳
**Status:** ⚠️ PARCIAL (existe mas quase vazio)  
**Impacto:** CRÍTICO - ACM rejeita automaticamente sem isto  
**Localização:** `README.md`  
**O que falta:**
- [ ] Título + descrição projeto
- [ ] Quick Start (comandos conda/pip/docker)
- [ ] Estrutura de diretórios explicada
- [ ] Resultados principais (3-5 linhas)
- [ ] Métricas chave (Detection Delay, Latency, etc.)
- [ ] Como citar projeto
- [ ] Referência para REPRODUCIBILIDADE.md

**Tamanho Esperado:** ~300-500 caracteres mínimo  
**Tempo Estimado:** 1-2 horas

---

## 🟠 ALTO (Paper + ACM Ready) - ~4-5 horas

### 6. **Criar scripts/statistical_analysis.py** ⏳
**Status:** ❌ FALTA (não existe)  
**Impacto:** ALTO - necessário para paper estatísticas  
**Localização:** `scripts/statistical_analysis.py`  
**O que falta:**
- [ ] Carregar `full_factorial_results.csv` (270 linhas)
- [ ] Calcular Mean ± Std por configuração
- [ ] Intervalo de Confiança 95% (IC95%)
- [ ] Teste Wilcoxon signed-rank (DET1 vs DET2)
- [ ] Tabelas LaTeX para paper
- [ ] Output: `results/metrics/full_factorial_summary.csv`

**Dependências:** Necessário ter 270 linhas primeiro (Task 1)  
**Tempo Estimado:** 1-2 horas

---

### 7. **Criar scripts/run_full_pipeline.py** ⏳
**Status:** ❌ FALTA (não existe)  
**Impacto:** ALTO - reproducibilidade end-to-end  
**Localização:** `scripts/run_full_pipeline.py`  
**O que falta:**
- [ ] Script orchestrator que executa: FE → Training → Factorial → Stats → Plots
- [ ] Logging de cada etapa
- [ ] Validação de outputs intermédios
- [ ] Erro handling

**Tempo Estimado:** 30 minutos

---

### 8. **Corrigir False-Positives em DET2 (D0)** ⏳
**Status:** ⚠️ CONHECIDO (não debugado)  
**Impacto:** ALTO - compromete validação  
**Localização:** `scripts/master_script.py` + `configs/config.yaml`  
**O que falta:**
- [ ] Analisar por que DET2 deteta drift em D0 (cenário sem drift)
- [ ] Problema documentado: D0 + DET2 = 19 detecções (esperado: 0)
- [ ] Ajustar ALPHA_KS (0.001 → 0.01) ou WINDOW_SIZE (20 → 30)
- [ ] Re-executar fatorial com parâmetros corrigidos
- [ ] Validar: D0 + DET2 agora → 0 detecções

**Tempo Estimado:** 1 hora

---

## 🟡 MÉDIO (Paper + Artifact) - ~7-10 horas

### 9. **Escrever Paper Draft** ⏳
**Status:** ❌ FALTA (não existe)  
**Impacto:** MÉDIO - necessário para submissão  
**Localização:** `paper/` (novo directório)  
**O que falta:**
- [ ] Criar `paper/main.md` ou `paper/main.tex`
- [ ] Secções obrigatórias:
  - Abstract (200 palavras)
  - Introduction (motivação Edge + Drift + PM)
  - Related Work (Concept Drift, Anomaly Detection)
  - Methods (Taxonomia drifts, detectores, adaptações)
  - Experimental Design (Fatorial, repetições, métricas)
  - Results (Tabelas, gráficos, análise estatística)
  - Discussion (Trade-offs, limitações, insights)
  - Conclusion + Future Work
  - References

**Tempo Estimado:** 4-6 horas

---

### 10. **Criar Plots Adicionais de Publicação** ⏳
**Status:** ✅ PARCIAL (2/5 plots existem)  
**Impacto:** MÉDIO - aumenta qualidade paper  
**Localização:** `scripts/generate_thesis_plots.py`  
**O que falta:**
- ✅ Fig1: Detection Delay (já existe)
- ✅ Fig2: Latency Comparison A1 vs A2 (já existe)
- [ ] Fig3: Recovery Time heatmap (Scenario × Detector × Adaptation)
- [ ] Fig4: FPR vs Detection Delay (Pareto front)
- [ ] Fig5: Diagrama setup hardware (foto + anotações)

**Tempo Estimado:** 1-2 horas

---

### 11. **Preparar Artifact Package para ACM** ⏳
**Status:** ❌ FALTA (não compilado)  
**Impacto:** MÉDIO - ACM badges  
**Localização:** Root directory (criar `artifact.zip`)  
**O que falta:**
- [ ] Criar `INSTALL.md` (instruções setup completas)
- [ ] Criar `RUN.md` (comandos exatos para reproduzir cada figura)
- [ ] Compilar `artifact.zip` contendo:
  - Código fonte (scripts/, src/)
  - Configs (configs/)
  - Dataset sample (<500 MB) ou traces
  - README.md, REPRODUCIBILIDADE.md, DATASET.md
  - Ficheiros de resultados amostra
  - Scripts regeneração plots
- [ ] Diagrama setup hardware (fotos/PDF)

**Tamanho Esperado:** <50 MB (sem dados raw)  
**Tempo Estimado:** 1-2 horas

---

## 📊 Tabela Resumida: Análise de Gaps

| # | Task | Ficheiro | Status | Prioridade | Tempo | Bloqueante? |
|---|------|----------|--------|-----------|-------|-----------|
| 1 | 5 Repetições Fatorial | `master_script.py` | ❌ | 🔴 CRÍTICO | 3-4h | ✅ SIM |
| 2 | requirements.txt | `env/` | ❌ | 🔴 CRÍTICO | 10m | ✅ SIM |
| 3 | environment.yml | `env/` | ❌ | 🔴 CRÍTICO | 10m | ✅ SIM |
| 4 | Dockerfile | `env/` | ❌ | 🔴 CRÍTICO | 15m | ✅ SIM |
| 5 | README.md | Root | ⚠️ Vazio | 🔴 CRÍTICO | 1-2h | ✅ SIM |
| 6 | statistical_analysis.py | `scripts/` | ❌ | 🟠 ALTO | 1-2h | ❌ NÃO |
| 7 | run_full_pipeline.py | `scripts/` | ❌ | 🟠 ALTO | 30m | ❌ NÃO |
| 8 | Fix DET2 FPR | `config.yaml` | ⚠️ | 🟠 ALTO | 1h | ❌ NÃO |
| 9 | Paper Draft | `paper/` | ❌ | 🟡 MÉDIO | 4-6h | ❌ NÃO |
| 10 | Plots Adicionais | `scripts/` | ✅ Parcial | 🟡 MÉDIO | 1-2h | ❌ NÃO |
| 11 | Artifact ACM | Root | ❌ | 🟡 MÉDIO | 1-2h | ❌ NÃO |

---

## ⏱️ Timeline de Prioridades

### **HOJE (Week 13 Deadline)**
🔴 **6-8 horas - Completar Tasks 1-5** (CRÍTICO para Week 15 gate)

### **PRÓXIMA SEMANA (Week 14)**
🟠 **4-5 horas - Completar Tasks 6-8** (Preparação paper)

### **SEMANA FINAL (Week 15)**
🟡 **7-10 horas - Completar Tasks 9-11** (Paper + Artifact finais)

---

## 📌 Notas Importantes

✅ **O que JÁ está feito:**
- Dados brutos congelados (D0-D5, ~1200 janelas cada)
- Features extraídas (6 ficheiros processados)
- Modelo baseline LOF treinado e validado
- Detectores DET0-2 implementados
- Adaptações A0-2 implementadas
- Fatorial 54 configs executado (1 rep apenas)
- 2 plots de publicação gerados
- Merge conflicts resolvidos
- 4 documentos de análise criados

❌ **O que FALTA é principalmente "finishing touches":**
- Repetições estatísticas (5 reps)
- Documentação final
- Dependências pinned
- Validação estatística completa
- Paper redação
- Packaging artifact

⚠️ **Problemas conhecidos a resolver:**
- DET2 tem false-positives em D0 (19 detecções esperado 0)
- Recovery Time suspeito (sempre 1.0)
- Código em `master_script.py` é acoplado (refactoring desejável mas não crítico)

---

## 🎯 Recomendação Final

**Sequência Ideal:**
1. **Hoje:** Tasks 1-5 (6-8h) → Pass Week 15 gate
2. **Próxima Semana:** Tasks 6-8 (4-5h) → Paper ready
3. **Semana Final:** Tasks 9-11 (7-10h) → Publication ready

**Esforço Total Estimado:** 15-23 horas (~2-3 semanas de trabalho focado)

**Prognóstico:** Com as correções, projeto atingirá 95-98% completude para submissão ACM.

---

**Documento criado:** 7 Maio 2026  
**Versão:** 1.0  
**Responsável:** GitHub Copilot (Análise Automática)
