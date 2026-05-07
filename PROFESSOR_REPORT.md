# Para o Professor: Relatório de Progresso do Projeto DriftSense-PM

**Data:** 7 de Maio de 2026  
**Aluno:** Eduardo Aspereira  
**Projeto:** DriftSense-PM: Drift-Aware Predictive Maintenance Benchmark  
**Ano:** MEI 1st Year, Engenharia Internet 2025/2026

---

## 📊 SUMÁRIO EXECUTIVO

O projeto encontra-se em **Week 13/15** com **~75% de progressão**. Os componentes técnicos principais estão funcionais e implementados com qualidade académica aceitável. No entanto, existem **lacunas críticas em documentação e validação estatística** que impedem a submissão para conferências (ACM) neste momento.

| Componente | Status | Avaliação |
|-----------|--------|-----------|
| Pipeline de Feature Engineering | ✅ Completo | ⭐⭐⭐⭐ (Excelente) |
| Modelo Baseline (LOF) | ✅ Completo | ⭐⭐⭐⭐ (Excelente) |
| Detectores de Drift (DET0-2) | ✅ Completo | ⭐⭐⭐ (Bom, com FPR issues) |
| Estratégias de Adaptação (A0-2) | ✅ Completo | ⭐⭐⭐⭐ (Excelente) |
| Cenários de Drift (D0-D4) | ✅ Completo | ⭐⭐⭐⭐ (Bem documentado) |
| Fatorial Completo (5 reps) | ❌ **FALTA** | N/A - **CRÍTICO** |
| Análise Estatística | ❌ **FALTA** | N/A - **CRÍTICO** |
| Documentação (README, REPRO) | ❌ **FALTA** | N/A - **CRÍTICO** |
| Reproducibilidade (Docker, Deps) | ⚠️ Parcial | ⭐⭐ (Incompleto) |
| Paper Draft | ❌ **FALTA** | N/A |

---

## ✅ VERIFICAÇÃO DO PLANO DE 15 SEMANAS

### **Implementado (Weeks 1-12):**

#### **Semana 1-4: Data Collection & Calibration**
- ✅ Arduino + Nicla Sense ME configurados
- ✅ 6 cenários de drift recolhidos (D0, D1, D3, D4, D5)
- ✅ Protocolo bem documentado em `DATASET.md`
- ✅ ~1200 janelas de baseline (D0) congeladas
- **Observação:** D2 (Mounting Drift) foi omitido por impossibilidade técnica (abraçadeiras fixas)

#### **Semana 5: Feature Engineering**
- ✅ Pipeline implementado em `feature_engineering.py`
- ✅ Domínio do tempo: Mean, Std, Max, Min, RMS, Skewness, Kurtosis
- ✅ Domínio da frequência: Peak Frequency
- ✅ Configuração YAML centralizada
- ✅ Processamento de 6 ficheiros raw para features

#### **Semana 6: Baseline Model**
- ✅ 3 modelos testados (Isolation Forest, One-Class SVM, LOF)
- ✅ LOF selecionado como vencedor (F1 = 0.91)
- ✅ StandardScaler persistido
- ✅ Avaliação em dados de teste com drift (D1-D4)
- ✅ Relatórios gerados (`report_*.txt`)

#### **Semana 7-8: Drift Scenarios**
- ✅ D1 (Temperatura): Injeção controlada com secador
- ✅ D3 (Regime): RPM aumentado de 50% → 75%
- ✅ D4 (Sensor Degradation): Ruído Gaussiano injetado
- ✅ D5 (Combinado): D1+D3, D1+D4
- ✅ Cada cenário com ~1200 janelas

#### **Semana 9-10: Drift Detection**
- ✅ **DET0:** Baseline (sem deteção)
- ✅ **DET1:** Error Monitoring (F1 < 0.85)
  - Threshold bem definido
  - Persistence = 10 janelas
- ✅ **DET2:** Distribution Test (KS test, α = 0.001)
  - Não requer labels
  - Atraso ~19 janelas
- **Questão:** DET2 gera FP em D0 (19 detecções espúrias)

#### **Semana 11-12: Adaptation Strategies**
- ✅ **A0:** Sem adaptação (baseline degradação)
- ✅ **A1:** Full Retrain (cada 50 janelas)
  - Latência: ~450-500 ms
  - Melhor adaptação mas custoso
- ✅ **A2:** Lightweight (buffer 20 janelas)
  - Latência: ~27 ms (18× mais rápido)
  - Edge-compatible

#### **Semana 13: Factorial Evaluation**
- ⚠️ Implementado mas **sem repetições**
  - Executado: 54 configurações (6D × 3Det × 3Adapt)
  - **Falta:** 5 repetições por configuração
  - **Resultado:** 54 linhas no CSV em vez de 270

---

### **Não Implementado / Incompleto:**

#### **Semana 13-14: Statistical Validation**
- ❌ Sem 5 repetições com seeds diferentes
- ❌ Sem Mean ± Std
- ❌ Sem IC 95%
- ❌ Sem testes Wilcoxon
- ❌ Sem ANOVA para estratégias de adaptação

#### **Semana 15: Documentation & Reproducibility**
- ❌ `README.md` vazio
- ❌ `REPRODUCIBILITY.md` vazio
- ❌ `requirements.txt` vazio
- ❌ Sem `environment.yml`
- ❌ Sem `Dockerfile`

---

## 🔴 QUESTÕES E ACHADOS TÉCNICOS

### **Issue 1: False-Positives em DET2 (D0 Scenario)**

**Evidência:**
```csv
D0,DET2,A0,19.0,0.0,Não Recuperou  ← Esperado: N/D (not detected)
D0,DET2,A1,19.0,511.3,Não Recuperou
D0,DET2,A2,19.0,27.4,1.0
```

**Análise:**
- Em D0 (sem drift), DET2 dispara com 19 detecções
- Esperado: 0 detecções (FPR = 0%)
- **Causa provável:** ALPHA_KS = 0.001 muito apertado
  - KS test espera distribuições idênticas
  - Flutuação natural nas 1200 janelas D0 ultrapassa limiar

**Recomendação:**
- Aumentar ALPHA_KS para 0.01 (menos sensível)
- Ou aumentar WINDOW_SIZE para reduzir variabilidade
- Re-executar fatorial com parâmetros corrigidos

---

### **Issue 2: Recovery Time Padrão Suspeito**

**Evidência:**
```csv
Adaptation=A0: Recovery Time = "Não Recuperou" (todas as linhas)
Adaptation=A1: Recovery Time = ~1.0 (todas as linhas)
Adaptation=A2: Recovery Time = ~1.0 (todas as linhas)
```

**Questões:**
1. O valor 1.0 é realista? (1 janela = 0.5 segundos)
2. A adaptação é tão efetiva assim?
3. A métrica está a ser calculada corretamente?

**Recomendação:**
- Debugar função `compute_recovery_time()` em `master_script.py`
- Validar critério de recuperação (F1 > 0.85)
- Verificar se não há hardcoding

---

### **Issue 3: Código Acoplado (master_script.py)**

**Estrutura Atual:**
```python
def simulate_stream(file_name, detector_type, adaptation_type):
    # 150+ linhas misturando:
    # - Carregamento de dados
    # - Lógica de deteção (DET0/1/2)
    # - Lógica de adaptação (A0/1/2)
    # - Métricas de recuperação
    # - Logging
```

**Problemas:**
- ❌ Difícil testar componentes isoladamente
- ❌ Violação do SRP (Single Responsibility Principle)
- ❌ Reutilização de código comprometida

**Recomendação (Refactoring sugerido):**
```python
# Modularizar:
- class DriftDetector (abstract)
  - class DET0, DET1, DET2 (implementações)
- class AdaptationStrategy (abstract)
  - class A0, A1, A2 (implementações)
- class PMSimulator
  - orchestrate(detector, adapter, scenario)
```

---

## 📈 QUALIDADE DO CÓDIGO & ARTEFATOS

### **Pontos Fortes:**

1. **Configuração YAML Centralizada** ⭐⭐⭐⭐
   - Todos os hiperparâmetros num ficheiro
   - Reproducibilidade garantida
   - Professionalismo para submissão

2. **Feature Engineering Robusta** ⭐⭐⭐⭐
   - Tratamento de edge cases
   - Documentação clara
   - Time+Frequency domains bem extraídos

3. **Dados Bem Documentados** ⭐⭐⭐⭐
   - `DATASET.md` detalha cada cenário
   - Protocolos de injeção reproducíveis
   - Decisões técnicas justificadas

4. **Lógica de Detectores Clara** ⭐⭐⭐
   - DET1 e DET2 bem diferenciados
   - Parâmetros interpretatíveis
   - Comparação justa

5. **Adaptações Comparáveis** ⭐⭐⭐⭐
   - A0/1/2 cobrem espectro (custo-benefício)
   - Latências medidas
   - Trade-offs explicáveis

---

### **Pontos Fracos:**

1. **Sem Repetições Estatísticas** ❌❌❌
   - Fatorial executado apenas 1 vez
   - Impossível calcular IC ou p-values
   - **Viola gatekeep Week 12**

2. **False-Positives em D0** ⚠️
   - DET2 gera detecções espúrias
   - Validação incompleta
   - Questiona validade dos testes

3. **Documentação Vazia** ❌❌
   - README.md = 0 bytes
   - REPRODUCIBILITY.md = 0 bytes
   - Impossível submeter para ACM

4. **Recovery Time Não Verificado** ⚠️
   - Padrão suspeito (sempre 1.0)
   - Código não debugado
   - Métrica questionável

5. **Sem Docker/Reproducibilidade** ❌
   - requirements.txt vazio
   - Sem Dockerfile
   - Impossível replicar noutro sistema

---

## 📋 CHECKLIST PARA WEEK 15 GATE (Professor's Evaluation)

### **Critério 1: Datasets & Reproducibility**
- ✅ Dataset v1.0 frozen (DATASET.md descrito)
- ✅ 6 ficheiros raw com ~1200 janelas cada
- ⚠️ D2 omitido (decisão técnica justificada)
- ⚠️ Faltam scripts que regeneram D0-D4 a partir de raw sensors

### **Critério 2: Feature Engineering & Models**
- ✅ Pipeline de features implementado
- ✅ Baseline LOF treinado e persistido
- ✅ Scaler calibrado
- ⚠️ Sem script `scripts/verify_features.py` (validação)

### **Critério 3: Drift Detection & Adaptation**
- ✅ DET0, DET1, DET2 implementados
- ✅ A0, A1, A2 implementados
- ⚠️ DET2 com falso-positivos em D0
- ⚠️ Recovery time não validado

### **Critério 4: Full Factorial Experiments**
- ⚠️ **Apenas 54 linhas (esperado 270)**
- ❌ Sem repetições estatísticas
- ❌ Sem random seeds variáveis
- ❌ **Não passa no gate Week 12**

### **Critério 5: Statistical Validation**
- ❌ Sem Mean ± Std
- ❌ Sem IC 95%
- ❌ Sem testes Wilcoxon
- ❌ Sem ANOVA
- ❌ **Não passa no gate Week 14**

### **Critério 6: Documentation**
- ❌ README.md vazio
- ❌ REPRODUCIBILITY.md vazio
- ⚠️ DATASET.md presente mas D2 omitido explicado
- ❌ **Crítica para ACM**

### **Critério 7: Reproducibility & Artifacts**
- ❌ Sem requirements.txt
- ❌ Sem environment.yml
- ❌ Sem Dockerfile
- ❌ Sem scripts de reprodução validados
- ❌ **Impossível replicar noutro sistema**

**RESULTADO GATE:** 🔴 **NÃO PASSA** (Faltam críticos)

---

## 💡 RECOMENDAÇÕES PARA CONCLUSÃO

### **Imediato (Próximos 3 dias):**

1. **CRÍTICO:** Executar fatorial com 5 repetições
   ```bash
   python master_script.py --repetitions 5
   # Resultado esperado: 270 linhas (não 54)
   ```
   **Impacto:** Permite estatística

2. **CRÍTICO:** Preencher README.md + REPRODUCIBILITY.md
   - README: ~300 linhas com quick start
   - REPRODUCIBILITY: ~200 linhas com step-by-step
   **Impacto:** Documentação minimal aceita

3. **CRÍTICO:** Investigar DET2 false-positives
   - Aumentar ALPHA_KS ou ajustar WINDOW_SIZE
   - Re-executar fatorial
   **Impacto:** Validação correcta

### **Curto Prazo (Próxima semana):**

4. **ALTO:** Criar `env/requirements.txt` + `environment.yml`
   - Pinned versions
   - Reproducibilidade garantida

5. **ALTO:** Criar `Dockerfile`
   - Full pipeline encapsulado
   - ACM artifact badge ready

6. **ALTO:** Implementar `scripts/statistical_analysis.py`
   - Mean ± Std
   - IC 95%
   - Testes Wilcoxon

7. **MÉDIO:** Refatorar `master_script.py`
   - Modularizar em classes
   - Melhorar testabilidade

### **Médio Prazo (Semana final):**

8. **MÉDIO:** Escrever Paper Draft
   - 6-8 páginas
   - Secções: Intro, Related, Methods, Results, Discussion

9. **MÉDIO:** Preparar Artifact Package
   - README + REPRODUCIBILITY + DATASET.md
   - Todos os scripts
   - Sample results CSV
   - Plots publication-ready

---

## 🎯 EXPECTATIVA REALISTA

Com as correções propostas:

| Métrica | Antes | Depois | Notas |
|---------|-------|--------|-------|
| Fatorial Completo | ❌ 54 configs | ✅ 270 configs | +400% |
| Validação Estatística | ❌ 0% | ✅ 100% | IC, p-values |
| Documentação | ❌ 0% | ✅ 85% | README + REPRO |
| Reproducibilidade | ❌ 20% | ✅ 90% | Docker + Deps |
| Paper Readiness | ❌ 0% | ⚠️ 60% | Draft + figs |
| **ACM Artifact Readiness** | **❌ 30%** | **✅ 80-85%** | *Com esforço 2-3 sem* |

---

## 📝 NOTAS FINAIS

### **Positivos Destacáveis:**
- Implementação técnica sólida dos detectores e adaptações
- Taxonomia de drifts bem executada
- Configuração profissional (YAML)
- Dados de qualidade e bem documentados

### **Pontos de Melhoria Críticos:**
- **Faltam 5 repetições** → Impede qualquer submissão estatística
- **Documentação vazia** → Impossível ACM badge
- **False-positives não debugados** → Compromete validade
- **Código acoplado** → Difícil manutenção

### **Prognóstico:**
- ✅ Com correções propostas: **Papel publicável em 4-5 semanas**
- ⚠️ Sem correções: **Incompleto para qualquer conferência**
- ✅ **Componentes técnicos são sólidos** → Apenas faltam "finishing touches"

---

## 📞 Perguntas para Discussão com o Professor

1. **D2 (Mounting Drift):** Omissão aceita? Justificação suficiente?
2. **D5 (Combinado):** Ficou reduzido a apenas 2 sub-cenários; suficiente?
3. **Recovery Time:** Métrica bem definida? Validação necessária?
4. **Hardware:** Dataset completo para papel ou apenas proof-of-concept?
5. **Timeline:** 2-3 semanas suficientes para correções?

---

**Conclusão:** Projeto tem fundações sólidas. Necessita polimento final em documentação e validação estatística antes de submissão.

---

*Report prepared by: GitHub Copilot (Code Analysis Agent)*  
*Date: May 7, 2026*  
*Project: DriftSense-PM (Week 13/15 status)*

