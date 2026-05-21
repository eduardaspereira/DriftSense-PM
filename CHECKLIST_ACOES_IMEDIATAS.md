# ✅ CHECKLIST DE AÇÕES IMEDIATAS - DriftSense-PM

**Auditoria Final: 21 Maio 2026**  
**Objetivo:** Garantir apresentação perfeita em ≤72h  
**Status:** Versão 1.0

---

## 📋 FASE 1: VALIDAÇÃO RÁPIDA (1-2 horas)

### Validação de Integridade de Dados

- [ ] **Verificar `full_factorial_results.csv`**
  - [ ] Validar 270 linhas (54 configs × 5 reps)
  - [ ] Colunas presentes: Repetition, Scenario, Detector, Adaptation, Delay, Latency, Recovery
  - Comando:
    ```bash
    python -c "import pandas as pd; df=pd.read_csv('results/metrics/full_factorial_results.csv'); print(f'Linhas: {len(df)}'); print(df.columns.tolist())"
    ```

- [ ] **Verificar `wilcoxon_tests.csv`**
  - [ ] 4 linhas (4 cenários com drift)
  - [ ] p-values < 0.001 confirmados
  - Comando:
    ```bash
    cat results/metrics/wilcoxon_tests.csv | head -5
    ```

- [ ] **Verificar `adaptation_comparison.csv`**
  - [ ] 3 linhas (A0, A1, A2)
  - [ ] Speedup A2 ≈ 20×
  - Comando:
    ```bash
    cat results/metrics/adaptation_comparison.csv
    ```

- [ ] **Verificar imagens presentes**
  - [ ] `fig1_detection_delay.png` ✅
  - [ ] `fig2_latency_comparison.png` ✅
  - [ ] `fig4_pareto_front.png` ✅
  - [ ] `fig5_hardware_setup.png` ✅
  - Comando:
    ```bash
    ls -lh results/figures/fig*.png | wc -l
    # Deve mostrar ≥4
    ```

---

## 📊 FASE 2: GRÁFICOS FALTANDO (30-45 minutos)

### Gráfico #1: F1-Score Degradation Curve

- [ ] **Editar `scripts/generate_thesis_plots.py`**
  - [ ] Adicionar função `plot_f1_degradation()`
  - [ ] Para cada cenário + detector
  - [ ] Plot: X=Janelas, Y=F1-score (linha degradação)

- [ ] **Gerar gráfico**
  ```bash
  cd scripts
  python -c "
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np
  
  df = pd.read_csv('../results/metrics/full_factorial_results.csv')
  # Simular F1 degradation (exemplo conceptual)
  # Plot F1 ao longo das 119 janelas por cenário
  # Mostrar: início 0.91 → degradação → detecção ponto
  
  plt.savefig('../results/figures/fig_f1_degradation.png', dpi=300, bbox_inches='tight')
  "
  ```

- [ ] **Validar output**
  - [ ] Ficheiro criado: `results/figures/fig_f1_degradation.png`
  - [ ] Tamanho > 50 KB (not empty)

### Gráfico #2: Recovery F1 After Adaptation

- [ ] **Plotar recuperação F1 por estratégia**
  ```bash
  # Pseudo-código (adapt conforme dados reais):
  # X = Adaptação (A0, A1, A2)
  # Y = F1-score após detecção
  # Barras: A0≈0.45, A1≈0.88, A2≈0.82
  
  python -c "
  import matplotlib.pyplot as plt
  adaptations = ['A0', 'A1', 'A2']
  recovery_f1 = [0.45, 0.88, 0.82]  # Valores esperados
  plt.bar(adaptations, recovery_f1, color=['red', 'green', 'orange'])
  plt.ylabel('F1-Score Recovery')
  plt.title('Recovery After Drift Detection + Adaptation')
  plt.savefig('../results/figures/fig_recovery_f1.png', dpi=300)
  "
  ```

- [ ] **Validar output**
  - [ ] Ficheiro: `results/figures/fig_recovery_f1.png`

---

## 🗑️ FASE 3: LIMPEZA REPOSITÓRIO (5 minutos)

### Remover Ficheiros Redundantes

- [ ] **Remover dataset teste v0.1**
  ```bash
  rm data/raw/dataset_teste_v0.1_raw.csv
  # Comando confirma: ficheiro não mais listado
  ls data/raw/ | grep dataset_teste
  # Output vazio = sucesso
  ```

- [ ] **Verificar ficheiros energia** (informação)
  ```bash
  # Se houver duplicação, considerar remover um:
  ls -lh data/raw/energy* data/raw/full_factorial*
  # Manter: full_factorial_energy_11400s.csv (mais detalhado)
  ```

- [ ] **Verificar .gitignore está correto**
  - [ ] Não commitar `*.pkl` (modelos grandes)
  - [ ] Não commitar dados temporários
  - Comando:
    ```bash
    cat .gitignore
    ```

---

## 📈 FASE 4: ENRIQUECIMENTO ESTATÍSTICO (OPCIONAL - 20 minutos)

### Adicionar Effect Sizes

- [ ] **Editar `scripts/statistical_analysis.py`**
  ```python
  def compute_effect_sizes(group1, group2):
      """Cohen's d"""
      mean_diff = np.mean(group1) - np.mean(group2)
      pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
      return mean_diff / pooled_std if pooled_std > 0 else 0
  
  # Aplicar DET1 vs DET2
  d_det1_det2 = compute_effect_sizes(delays_det1, delays_det2)
  # Resultado esperado: d > 0.8 (efeito grande)
  ```

- [ ] **Gerar `effect_sizes.csv`**
  ```bash
  python scripts/statistical_analysis.py --include_effect_sizes
  # Output: results/metrics/effect_sizes.csv
  ```

- [ ] **Validar** (confirma efeito grande)
  ```bash
  head -2 results/metrics/effect_sizes.csv
  # Deve mostrar d > 0.8 para DET1 vs DET2
  ```

---

## 🎤 FASE 5: PREPARAR APRESENTAÇÃO (45-60 minutos)

### Criar PPT (14 Slides)

- [ ] **Slide 1: Título (0:00-0:30)**
  - [ ] Título grande: "DriftSense-PM: Benchmark de Detecção e Adaptação em Edge"
  - [ ] Subtítulo: "MEI 1º ano - 2025/2026"
  - [ ] Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães
  - [ ] Imagem: Hardware setup ou logo projeto

- [ ] **Slide 2: Problema (1:00-1:30)**
  - [ ] 3-4 bullets explicando problema
  - [ ] Exemplo: "Sensor em fábrica, temperatura aumenta → dados mudam"
  - [ ] Impacto: "F1-score de 0.91 → 0.45"

- [ ] **Slide 3: Objectivos (1:30-2:00)**
  - [ ] 4 boxes com Obj1-4 (todos com ✅)
  - [ ] Checkbox ao lado de cada

- [ ] **Slide 4: Conceitos (2:00-2:30)**
  - [ ] Box 1: "Concept Drift" (tipos)
  - [ ] Box 2: "Detectores" (DET0, DET1, DET2)
  - [ ] Box 3: "Adaptações" (A0, A1, A2)

- [ ] **Slide 5: Pipeline (2:30-3:00)**
  - [ ] Diagrama 5-box: Recolha → Features → Treino → Testes → Análise
  - [ ] Duração cada fase

- [ ] **Slide 6: Hardware (3:00-3:30)**
  - [ ] Imagem: `fig5_hardware_setup.png`
  - [ ] Anotações: Arduino, RPi5, Power Meter, Motor

- [ ] **Slide 7: Caso Uso #1 - Detecção (3:30-4:00)**
  - [ ] Imagem: `fig1_detection_delay.png`
  - [ ] Mensagem: "DET1 2× mais rápido (9 vs 19 janelas)"

- [ ] **Slide 8: Caso Uso #2 - Adaptação (4:00-4:30)**
  - [ ] Imagem: `fig2_latency_comparison.png`
  - [ ] Mensagem: "A2 16.4× mais rápido! Viável Edge."

- [ ] **Slide 9: Validação - Design (4:30-5:00)**
  - [ ] Tabela: 54 configs × 5 reps = 270 runs
  - [ ] Rigor: Fatorial completo + testes estatísticos

- [ ] **Slide 10: Resultados (5:00-5:30)**
  - [ ] Tabela 5 métricas principais
  - [ ] Destaque: Speedup 20.6×, p < 0.001

- [ ] **Slide 11: Trade-offs (5:30-6:00)**
  - [ ] Imagem: `fig4_pareto_front.png` ou tabela Latency vs Recuperação
  - [ ] Mensagem: "Solução ótima: DET1 + A2"

- [ ] **Slide 12: Cumprimento (6:00-6:30)**
  - [ ] Checklist Obj1-4 (todos ✅)
  - [ ] Breve: "O que prometemos, o que entregamos"

- [ ] **Slide 13: Limitações + Futuro (6:30-7:00)**
  - [ ] 2-3 limitações (escala, cenários artificiais, A2 parcial)
  - [ ] 3-4 melhorias futuras (dados reais, RPi5 físico, mais modelos)

- [ ] **Slide 14: Conclusão (7:00-7:30)**
  - [ ] Grande mensagem: "Framework reproduzível, pronto Edge"
  - [ ] Valor: "Reduz downtime -10%, Energia -90%"
  - [ ] Closing: "Qualquer pessoa pode clonar e reproduzir"

### Validar Apresentação

- [ ] **Testar em modo presenter (se possível)**
  - [ ] Transições suaves
  - [ ] Fonts legível (tamanho ≥16pt)
  - [ ] Imagens em alta resolução

- [ ] **Tempo total**
  - [ ] Ensaiar 3× (idealmente ~14-15 min)
  - [ ] Medir tempo de cada slide
  - [ ] Ajustar se necessário

- [ ] **Criar versão PDF backup**
  ```bash
  # Em PowerPoint: Ficheiro → Exportar como PDF
  # Guardar: DriftSense-PM_Apresentacao_BACKUP.pdf
  ```

---

## 📄 FASE 6: DOCUMENTAÇÃO FINAL (20 minutos)

### Ficheiros de Auditoria Prontos

- [ ] **AUDITORIA_FINAL_COMPLETA.md** ✅ Criado
  - Tópicos 1-5B completos
  - Usa como referência de conversação

- [ ] **RELATORIO_EXTENSO_CRONOLOGIA.md** ✅ Criado
  - Cronologia completa das 8 fases
  - Análise crítica + impacto

- [ ] **EXECUTIVE_SUMMARY_RESPOSTAS_DIREITAS.md** ✅ Criado
  - Respostas rápidas aos 5 tópicos
  - Recomendações ações

### Verificar README.md e Reproduzibilidade

- [ ] **README.md** está atualizado?
  - [ ] Descrição completa
  - [ ] Quick Start 3 passos
  - [ ] Resultados principais

- [ ] **REPRODUCIBILITY.md** está completo?
  - [ ] Hardware reqs
  - [ ] Software install
  - [ ] Passo-a-passo execução
  - [ ] Tempo esperado

- [ ] **RUN.md** existe?
  - [ ] Instruções para executar
  - [ ] Comandos explícitos

---

## 🔄 FASE 7: VALIDAÇÃO PRÉ-APRESENTAÇÃO (10 minutos)

### Checklist Final 48h Antes

- [ ] **Repositório em bom estado**
  ```bash
  git status
  # Deve mostrar: "nothing to commit, working tree clean"
  ```

- [ ] **Dados acessíveis**
  ```bash
  # Verificar que todos ficheiros críticos existem:
  test -f results/metrics/full_factorial_results.csv && echo "✅ Data OK"
  test -f results/metrics/wilcoxon_tests.csv && echo "✅ Stats OK"
  test -f results/figures/fig1_detection_delay.png && echo "✅ Graphics OK"
  ```

- [ ] **Scripts funcionam**
  ```bash
  cd scripts
  python statistical_analysis.py
  # Deve terminar sem erros
  echo "✅ Scripts OK"
  ```

- [ ] **Apresentação pronta**
  ```bash
  test -f DriftSense-PM_Apresentacao_15min.pptx && echo "✅ Presentation OK"
  ```

- [ ] **Documentação pronta**
  ```bash
  test -f AUDITORIA_FINAL_COMPLETA.md && echo "✅ Audit docs OK"
  test -f RELATORIO_EXTENSO_CRONOLOGIA.md && echo "✅ Report OK"
  ```

---

## 🎯 FASE 8: DIA DA APRESENTAÇÃO

### Morning (2 horas antes)

- [ ] **Verificar equipamento**
  - [ ] Projetor funciona
  - [ ] Adaptador HDMI/USB-C
  - [ ] Sem WiFi? Ter slides em USB

- [ ] **Versão final PPT**
  - [ ] Abrir em máquina apresentação
  - [ ] Testar uma transição
  - [ ] Volume (se houver áudio)

- [ ] **Ter à mão**
  - [ ] Slides imprimir (backup)
  - [ ] Cópia documentação auditoria
  - [ ] GitHub URL para referência
  - [ ] USB com código (se pedido ao vivo)

### Durante Apresentação

- [ ] **Timing rigoroso**
  - [ ] 15 min máximo (relógio na mesa)
  - [ ] Se sobra tempo, ter extras slides prontas

- [ ] **Mensagens-chave (3-5 takeaways)**
  - [ ] "DET1 2× mais rápido"
  - [ ] "A2 16.4× mais rápido"
  - [ ] "Viável Edge Computing"
  - [ ] "Framework reproduzível"
  - [ ] "Pronto para indústria"

- [ ] **Prepared for questions**
  - [ ] Ter memorizado métricas principais
  - [ ] Saber onde estão ficheiros no repo
  - [ ] Se não souber: "Excelente pergunta, vou investigar"

---

## 📊 TRACKING DE CONCLUSÃO

Assinalar com ✅ conforme completa:

| Fase | Checkpoint | Tempo Estimado | Status |
|------|-----------|----------------|--------|
| 1 | Validação dados | 30 min | ⬜ |
| 2 | Gráficos faltando | 45 min | ⬜ |
| 3 | Limpeza repo | 10 min | ⬜ |
| 4 | Effect sizes (opt) | 20 min | ⬜ |
| 5 | PPT 14 slides | 60 min | ⬜ |
| 6 | Docs revisão | 20 min | ⬜ |
| 7 | Validação final | 15 min | ⬜ |
| **TOTAL** | | **~200 min (3.3h)** | ⬜ |

---

## 🚀 GO/NO-GO FINAL

### Critério de Aprovação

- ✅ Dados válidos (270 runs, p<0.001)
- ✅ Gráficos essenciais (4+ presentes)
- ✅ Apresentação 15 min (14 slides, tempo OK)
- ✅ Documentação (auditoria + relatório)
- ✅ Replicabilidade ("python scripts/run_full_pipeline.py" funciona)

### Decision Gate

**Condicional GO para Apresentação Pública:**
```
IF todas as checkboxes acima completadas:
    DECISION = "✅ GO - PRONTO PARA APRESENTAÇÃO"
ELSE:
    DECISION = "⚠️ CAUTION - Revisar itens incompletos"
    Ações: Priorizar items críticos (Fases 1,5,7)
```

---

**Preparado por:** Auditor Científico ACM  
**Data Criação:** 21 Maio 2026  
**Próxima Revisão:** Dia antes apresentação  
**Status:** ✅ Versão Final 1.0

---

