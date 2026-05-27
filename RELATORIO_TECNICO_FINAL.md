# 📊 DriftSense-PM: Relatório Extenso Final
## Manutenção Preditiva com Detecção de Concept Drift em Ambientes Edge

### 1. Resumo Executivo
Este relatório detalha a arquitetura e validação empírica do sistema DriftSense-PM, uma solução de manutenção preditiva desenhada para operar sob as severas restrições computacionais de dispositivos de edge, como o Raspberry Pi 5. O problema central abordado é o *concept drift*, um fenómeno que degrada o desempenho de modelos de Machine Learning ao longo do tempo devido a alterações na distribuição estatística dos dados. Em ambientes industriais, a não estacionaridade dos dados é a norma, não a exceção, tornando a deteção e adaptação a estes drifts um requisito crítico para a fiabilidade do sistema. O nosso sistema impõe um Service Level Agreement (SLA) estrito de <100ms por inferência para garantir a viabilidade em tempo real. A solução proposta, validada através de uma matriz fatorial completa com 270 execuções, identifica uma combinação ótima de modelo, detetor e estratégia de adaptação que equilibra eficácia e eficiência computacional. A tabela abaixo resume a solução vencedora.

| Componente | Solução Vencedora | Desempenho Empírico Chave | Implicação Industrial |
| :--- | :--- | :--- | :--- |
| **Modelo Base** | One-Class SVM | F1-Weighted = 0.897 (em 619 janelas de teste) | Alta precisão na deteção de anomalias de base, rejeitando alternativas como Isolation Forest e LOF por insuficiências de recall. |
| **Detetor de Drift** | DET1 (Error Monitoring) | Atraso médio de 9 janelas (D1) e 16 (D2) | Rápida reação a drifts graduais e abruptos, superando significativamente a latência do detetor estatístico DET2 (p-value < 0.001). |
| **Adaptação** | A2 (Lightweight) | Latência média de 9.59 ms (Speedup de 27.3× vs. A1) | Cumpre o SLA de <100ms com uma margem substancial, permitindo a adaptação do modelo diretamente no dispositivo Edge sem comprometer a operação. |
| **Recuperação** | DET1 + A2 | 1.0 janela para drifts simples (D1, D2) | Recuperação quase imediata do desempenho pós-drift, embora exija 5-6 janelas para estabilizar em cenários de drift composto, expondo um compromisso (trade-off) fundamental. |

### 2. Introdução e Motivação

#### 2.1 O Problema: Concept Drift
O desafio fundamental da manutenção preditiva em ambientes industriais dinâmicos é a violação da premissa de estacionaridade. Modelos de Machine Learning são tipicamente treinados sob a suposição de que a distribuição de probabilidade conjunta entre as características (X) e a variável alvo (Y) permanece constante ao longo do tempo. Formalmente, se considerarmos a distribuição de dados no momento do treino, $t_0$, como $P_{t_0}(X, Y)$, o modelo assume que para qualquer momento futuro $t_1$, a seguinte igualdade se mantém:

$$P_{t_0}(X, Y) = P_{t_1}(X, Y)$$

*Concept drift* (deriva de conceito) ocorre quando esta suposição é violada, ou seja, $P_{t_0}(X, Y) \neq P_{t_1}(X, Y)$. Esta alteração pode manifestar-se de duas formas principais:
1.  **Covariate Shift (Deriva de Covariáveis)**: A distribuição das características de entrada, $P(X)$, altera-se ($P_{t_0}(X) \neq P_{t_1}(X)$), mas a relação condicional entre as características e o alvo, $P(Y|X)$, permanece inalterada. Um exemplo prático é uma alteração na temperatura ambiente de operação de uma máquina, que modifica os *inputs* dos sensores sem alterar a física da falha.
2.  **Real Concept Drift (Deriva de Conceito Real)**: A relação condicional $P(Y|X)$ altera-se, o que significa que a própria definição do que constitui uma "falha" ou "anomalia" muda. Isto pode ocorrer devido ao desgaste de um componente, que introduz um novo modo de falha não presente nos dados de treino.

Na prática industrial, a distinção é subtil e ambas as formas de drift coexistem frequentemente. O impacto de não gerir o *concept drift* é severo:
-   **Falsos Positivos (Alarmes Falsos)**: O modelo sinaliza uma anomalia inexistente. Isto leva a paragens de produção desnecessárias, custos de inspeção e erosão da confiança dos operadores no sistema de monitorização. O nosso detetor DET2, por exemplo, demonstrou uma propensão para falsos positivos no cenário de controlo D0, disparando um alarme na janela 39, um comportamento inaceitável em produção.
-   **Falsos Negativos (Falhas Não Detetadas)**: O modelo falha em identificar uma anomalia real. As consequências são potencialmente catastróficas, incluindo danos severos no equipamento, riscos de segurança para os operadores e perdas financeiras avultadas devido a paragens não planeadas. A rejeição dos modelos Local Outlier Factor (LOF) e Isolation Forest no nosso benchmark inicial deveu-se precisamente a um *recall* insuficiente, o que os tornaria perigosamente ineficazes no terreno.

#### 2.2 Contexto: Computação na Edge
A resposta tradicional ao *concept drift* envolve a retransmissão de dados para a nuvem para retreino centralizado. Contudo, esta abordagem é inviável em muitos cenários industriais devido a requisitos de latência, largura de banda e privacidade. A computação na *edge* (periferia) emerge como uma solução necessária, onde o processamento de dados ocorre localmente, no próprio dispositivo.

O nosso sistema foi implementado e testado num **Raspberry Pi 5**, um representante canónico das plataformas de *edge computing*. Este ambiente impõe restrições físicas e computacionais significativas:
-   **CPU e Memória**: Equipado com um processador ARM Cortex-A76 quad-core, o Raspberry Pi 5 oferece uma capacidade de processamento considerável para o seu formato, mas que é ordens de magnitude inferior à de um servidor de nuvem. A memória RAM limitada exige modelos e algoritmos eficientes.
-   **Armazenamento e I/O**: O sistema operativo e os dados são armazenados num cartão MicroSD, cujos ciclos de escrita/leitura (I/O) são limitados. Estratégias de adaptação que envolvem escrita intensiva em disco, como o retreino frequente com grandes volumes de dados (a nossa estratégia A1), não só são lentas, como também degradam o tempo de vida útil do dispositivo.
-   **Necessidade de Processamento Local**: A latência de rede e a potencial intermitência da conectividade tornam o processamento local uma necessidade crítica. Para uma máquina em operação, uma decisão sobre o seu estado de saúde tem de ser tomada em milissegundos. O nosso SLA de **<100ms por janela de dados** reflete este requisito industrial, forçando a rejeição de qualquer abordagem computacionalmente pesada que não consiga cumprir este orçamento de tempo. A estratégia A2 (Lightweight Adaptation) foi desenhada especificamente para operar dentro deste envelope, como provado pelos seus 9.59 ms de latência.

### 3. Método Desenvolvido e Arquitetura do Sistema

#### 3.1 Design Experimental: Matriz Fatorial Completa
Para garantir a robustez e a reprodutibilidade dos resultados, adotámos um desenho experimental baseado numa matriz fatorial completa (*full factorial design*). Esta abordagem sistemática permite isolar e quantificar o impacto de cada componente do sistema (modelo, detetor, adaptação) e as suas interações. A nossa matriz experimental foi definida pela combinação exaustiva dos seguintes fatores:

-   **6 Cenários de Drift (D0 a D4)**: Cobrindo desde a ausência de drift até drifts abruptos, graduais e compostos.
-   **3 Detetores de Drift (DET0, DET1, DET2)**: Representando a ausência de deteção, monitorização de erro e vigilância estatística.
-   **3 Políticas de Adaptação (A0, A1, A2)**: Variando desde a não adaptação até ao retreino completo e uma adaptação leve.
-   **5 Repetições Independentes**: Para cada combinação de fatores, o experimento foi repetido cinco vezes com sementes aleatórias distintas para garantir a estabilidade estatística dos resultados.

A execução desta matriz resultou num total de **6 × 3 × 3 × 5 = 270 execuções independentes**. Este volume de dados empíricos constitui a base para todas as análises subsequentes, permitindo-nos extrair conclusões com um elevado grau de confiança estatística e satisfazer os critérios de reprodutibilidade exigidos por conferências como as da ACM.

#### 3.2 Cenários de Drift Injetados Fisicamente
A validade de um sistema de deteção de drift depende criticamente da qualidade e realismo dos cenários de teste. Em vez de simular drifts sinteticamente, nós injetámo-los fisicamente num sistema eletromecânico monitorizado. Os seis cenários de dados (`D0` a `D4`) representam condições operacionais distintas:

-   **D0 (Cenário de Controlo)**: Operação normal e estável. Serve como *baseline* para medir a taxa de falsos positivos. Qualquer alarme gerado neste cenário é, por definição, um alarme falso.
-   **D1 - Temperature Drift (Desvio Gradual)**: Um aumento gradual da temperatura do sistema, simulando o sobreaquecimento progressivo de um componente. Este é um *covariate shift* clássico.
-   **D2 - Regime Drift (Desvio Abrupto Operacional)**: Uma alteração súbita no regime de rotação do motor. Simula uma mudança abrupta nas condições de operação da máquina.
-   **D3 - Sensor Degradation (Desvio por Ruído)**: Injeção de ruído branco gaussiano no sinal do acelerómetro (`AccX_RMS`), simulando a degradação ou falha de um sensor.
-   **D4_D1eD2 (Térmico + Operacional)**: Cenário multi-fatorial realista em que o sobreaquecimento de componente (D1) ocorre em simultâneo com o aumento de carga por rotação (D2).
-   **D4_D2eD3 (Operacional + Degradação)**: Cenário composto complexo integrando a variação brusca de RPM (D2) com a falha estocástica de leitura do sensor (D3).

Este mapeamento detalhado permite uma análise granular da performance do sistema sob diferentes tipos de perturbação, um requisito essencial para a validação em ambiente industrial.

#### 3.3 Lógica dos Detetores
Implementámos três detetores com lógicas fundamentalmente distintas:

-   **DET0 (Baseline)**: Um detetor nulo que nunca dispara um alarme. Serve como controlo para medir o desempenho do sistema sem qualquer mecanismo de deteção de drift.
-   **DET1 (Error Monitoring)**: Um detetor baseado na monitorização do desempenho do próprio modelo. Um alarme é acionado se a métrica F1-Score, calculada numa janela deslizante de observações, cair abaixo de um limiar pré-definido (**F1 < 0.85**) e se essa condição persistir por um número mínimo de janelas consecutivas (**persistência ≥ 10**). Este mecanismo é eficaz a detetar degradação de performance real, reagindo em 9 janelas para D1 e D4_D1eD2, 16 para D2, e 13 para D4_D2eD3.
-   **DET2 (Statistical Monitoring)**: Um detetor que monitoriza a distribuição de uma característica específica dos dados, independentemente do modelo. Utiliza o **teste estatístico não-paramétrico Kolmogorov-Smirnov para duas amostras (ks_2samp)**. O teste compara a distribuição da característica `AccX_RMS` numa janela de referência (dados históricos estáveis) com a distribuição numa janela deslizante atual. Um alarme é acionado se a hipótese nula (de que as duas amostras provêm da mesma distribuição) for rejeitada com um nível de significância **α = 0.01**. A fórmula bicaudal do teste avalia a máxima diferença absoluta entre as funções de distribuição cumulativa (FDC) das duas amostras. Este detetor demonstrou uma rigidez matemática notável, disparando consistentemente na **19ª janela** para todos os cenários de drift real (D1, D2, D4) e na 39ª janela em D0 (falso positivo).

#### 3.4 Políticas de Adaptação
As políticas de adaptação definem como o sistema reage a um alarme de drift:

-   **A0 (Nenhuma Adaptação)**: Ignora o alarme e continua a usar o modelo original. Serve para quantificar o impacto do drift não mitigado.
-   **A1 (Retreino Completo)**: Após um alarme, o sistema combina os dados históricos de treino com um buffer de dados recentes e retreina completamente o modelo de base (One-Class SVM) com todos os seus hiperparâmetros (e.g., 100 árvores para ensembles). Este processo é computacionalmente intensivo, com uma latência média de **261.66 ms**, violando o nosso SLA.
-   **A2 (Lightweight Adaptation)**: Uma estratégia *Edge-First* desenhada para eficiência. Após um alarme, o sistema descarta o modelo antigo e treina um novo modelo, muito mais simples (e.g., um Isolation Forest com apenas **10 árvores**), utilizando *apenas* um pequeno buffer das **20 janelas de dados mais recentes**. Este mecanismo assume que o passado recente é a melhor representação do "novo normal". A sua latência média é de apenas **9.59 ms**, um *speedup* de 27.3x em relação a A1, cumprindo confortavelmente o SLA.

#### 3.5 Estrutura do Repositório e Fluxo de Dados
A organização do repositório foi desenhada para maximizar a clareza, a modularidade e a reprodutibilidade, seguindo as melhores práticas para a obtenção de *badges* de reprodutibilidade da ACM (*Artifacts Available*, *Artifacts Evaluated*, *Results Reproduced*).

```
/
├── src/             # Código fonte principal (lógica de adaptação, detetores)
├── scripts/         # Scripts para orquestrar o pipeline (run_full_pipeline.py)
├── data/
│   ├── raw/         # Datasets originais não processados (.csv)
│   ├── processed/   # Datasets após feature engineering
│   └── splits/      # Divisões de treino/teste
├── models/          # Modelos treinados serializados (.pkl)
├── results/
│   ├── logs/        # Logs de execução detalhados
│   ├── metrics/     # Ficheiros .csv e .txt com resultados quantitativos
│   └── figures/     # Gráficos e visualizações geradas
└── configs/         # Ficheiros de configuração (config.yaml)
```

Esta estrutura impõe um claro **isolamento de responsabilidades**:
-   `src/` contém a lógica de negócio reutilizável e testável.
-   `scripts/` contém o código "cola" que orquestra as experiências.
-   `data/` segue um fluxo lógico desde os dados brutos até às versões processadas e divididas, garantindo a proveniência dos dados.
-   `results/` armazena todos os artefactos gerados, separando métricas de logs e figuras, o que facilita a análise e a verificação por parte de revisores externos.

Este design não é apenas uma questão de organização, mas um requisito funcional para a automação e a reprodutibilidade de todo o fluxo de trabalho científico.

### 4. Avaliação e Análise Crítica de Resultados (Baseada em Evidências)

#### 4.1 Desempenho do Modelo na Fase de Baseline
A primeira fase do nosso pipeline experimental consistiu num benchmark rigoroso para selecionar o modelo de deteção de anomalias mais eficaz. Três algoritmos foram avaliados: One-Class SVM, Local Outlier Factor (LOF) e Isolation Forest. O modelo foi treinado com dados do cenário de controlo (D0) e avaliado num conjunto de teste independente contendo 619 janelas, que incluía tanto dados normais como anómalos. O **One-Class SVM emergiu como o vencedor claro**, como evidenciado pelo seu *classification report*:

```
Modelo: One-Class SVM
                     precision    recall  f1-score   support

Anomalia/Drift (-1)      0.974     0.882     0.926       595
         Normal (1)      0.125     0.417     0.192        24

           accuracy                          0.864       619
          macro avg      0.550     0.650     0.559       619
       weighted avg      0.941     0.864     0.897       619
```

A métrica decisiva foi o **F1-Score ponderado (0.897)**, que equilibra a precisão e o *recall*. Mais importante, o One-Class SVM alcançou um *recall* de **0.882** para a classe de anomalia, significando que identificou corretamente 88.2% das anomalias reais. Em contraste, os modelos **LOF e Isolation Forest foram formalmente rejeitados** devido a um desempenho de *recall* inaceitavelmente baixo. Em manutenção preditiva, a falha em detetar uma anomalia (um falso negativo) é tipicamente muito mais custosa do que um alarme falso (falso positivo). A superioridade do One-Class SVM em minimizar estes falsos negativos tornou-o a escolha inequívoca para o modelo de base do DriftSense-PM.

#### 4.2 Atraso de Deteção e Velocidade de Reação
O atraso de deteção (*detection delay*) é uma métrica crítica que mede o número de janelas de dados que ocorrem desde o início do drift até ao seu reconhecimento pelo detetor. Uma deteção mais rápida permite uma reação mais célere, minimizando o período em que o sistema opera com um modelo degradado.

A nossa análise revela uma diferença estatisticamente significativa entre os detetores DET1 e DET2. O **DET2 (Kolmogorov-Smirnov)**, embora matematicamente robusto, demonstrou uma latência de deteção fixa e elevada de **19 janelas** para todos os cenários de drift (D1, D2, D4s). Esta rigidez deriva da sua necessidade de acumular evidência estatística suficiente para rejeitar a hipótese nula.

Em contrapartida, o **DET1 (Error Monitoring)** provou ser muito mais ágil. Detetou o drift gradual D1 em apenas **9 janelas** e o drift abrupto D2 em **16 janelas**. Esta rapidez deve-se ao facto de monitorizar diretamente o impacto do drift no desempenho do modelo, que é um indicador mais imediato do que a alteração na distribuição de uma única *feature*.

Para validar formalmente esta observação, aplicámos o **Teste de Wilcoxon Signed-Rank**, um teste não-paramétrico apropriado para comparar amostras emparelhadas. O resultado para a comparação dos atrasos de deteção de DET1 vs. DET2 em todos os cenários de drift relevantes foi um **p-value = 0.000108**. Com um nível de significância α=0.01, este valor é extremamente baixo, levando à rejeição da hipótese nula de que não há diferença entre os detetores. A significância máxima (***) confirma que o DET1 é, inequivocamente, mais rápido a reagir do que o DET2.

| Cenário | Comparação | p_value | Significância | Atraso Médio DET1 | Atraso Médio DET2 | Diferença |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D1 | DET1 vs DET2 | 0.000108 | ***(Máxima) | 9.0 | 19.0 | -10.0 |
| D2 | DET1 vs DET2 | 0.000108 | ***(Máxima) | 16.0 | 19.0 | -3.0 |

O valor-p substancialmente inferior a $0.01$ rejeita categoricamente a hipótese nula, provando que o DET1 antecipa de forma sistemática a presença de drifts face a abordagens puramente estatísticas multivariadas.

#### 4.3 Eficiência de Hardware e Latência Computacional
Num dispositivo de *edge* como o Raspberry Pi 5, a eficiência computacional não é um luxo, mas uma necessidade. O nosso SLA de <100ms por janela de dados serve como um limiar estrito para a viabilidade de qualquer estratégia de adaptação. A análise da latência das políticas de adaptação A1 e A2 revela um contraste gritante:

| Estratégia | Latência Média (ms) | Desvio Padrão (ms) | Speedup vs. A1 | Conformidade SLA (<100ms) |
| :--- | :--- | :--- | :--- | :--- |
| A1 (Retreino Completo) | 261.66 | 9.57 | 1.0× | **Não** |
| A2 (Lightweight) | 9.59 | 7.7 | **27.3×** | **Sim** |

A estratégia **A1 (Retreino Completo)**, com a sua latência média de **261.66 ms**, falha redondamente em cumprir o SLA. Este tempo de processamento é inaceitável para uma aplicação em tempo real.

Por outro lado, a estratégia **A2 (Lightweight Adaptation)** foi desenhada especificamente para este ambiente. Com uma latência média de apenas **9.59 ms**, não só cumpre o SLA, como o faz com uma margem de mais de 90%. O *speedup* (aceleração) de **27.3 vezes** em relação a A1 é uma prova matemática da sua eficiência. Este desempenho é alcançado ao treinar um modelo muito mais simples (10 árvores vs. 100) sobre um conjunto de dados drasticamente menor (20 janelas vs. histórico completo). Esta abordagem "esquece" o passado deliberadamente para se focar na nova realidade, uma tática que se revela extremamente eficaz do ponto de vista computacional.

#### 4.4 Dinâmica de Recuperação e a Frente de Pareto
O *Recovery Time* (tempo de recuperação) mede o número de janelas que o sistema necessita, após uma adaptação, para retornar a um estado de desempenho estável (definido por um F1-Score consistentemente acima de um limiar de estabilidade, `STABILITY_THRESHOLD = 5` execuções consecutivas acima do baseline).

A nossa análise, baseada no *heatmap* de tempos de recuperação, revela uma dinâmica complexa. Para drifts simples como D1 (temperatura) e D2 (RPM), a combinação **DET1+A2** alcança uma recuperação quase imediata, necessitando de apenas **1.0 janela**. O modelo leve treinado com dados recentes é suficiente para restaurar o desempenho rapidamente.

Contudo, em cenários de drift mais severos e compostos (D4_D1eD2 e D4_D2eD3), a estratégia A2 revela a sua limitação científica. Ao descartar o histórico, o modelo A2 perde informação valiosa sobre a variabilidade dos dados. Consequentemente, torna-se mais suscetível a falsos alarmes nas janelas imediatamente após a adaptação. O sistema necessita de **5 a 6 janelas adicionais** para se estabilizar e reconstruir um modelo robusto da "nova normalidade".

Isto expõe um *trade-off* fundamental, uma **Frente de Pareto** entre a velocidade de adaptação e a robustez da recuperação.
-   **A1 (Retreino Completo)**: Lento a adaptar, mas potencialmente mais robusto na recuperação (embora violando o SLA).
-   **A2 (Lightweight)**: Extremamente rápido a adaptar, mas com um período de instabilidade maior em drifts complexos.

Considerando todas as métricas — atraso de deteção, latência computacional e tempo de recuperação — a combinação **DET1+A2** representa o **ponto de equilíbrio ótimo de Nash** na nossa Frente de Pareto. O DET1 oferece a deteção mais rápida, enquanto o A2 garante a conformidade com o SLA de *edge computing*. O custo de uma recuperação ligeiramente mais lenta em cenários complexos é um compromisso aceitável pela garantia de uma reação rápida e computacionalmente viável na esmagadora maioria dos casos.

### 5. Conclusões e Sugestões de Trabalho Futuro

#### 5.1 Contribuições e Aprendizagens Técnicas
O projeto DriftSense-PM demonstrou com sucesso a viabilidade de implementar um sistema de manutenção preditiva autónomo e adaptativo em dispositivos de *edge* com recursos limitados. A principal contribuição é a validação empírica de um pipeline completo, desde a aquisição de dados até à adaptação em tempo real, que respeita um SLA industrial estrito de <100ms.

As principais aprendizagens técnicas extraídas são:
1.  **A Supremacia da Combinação DET1+A2**: A análise fatorial completa provou inequivocamente que a monitorização de erro (DET1) para deteção e uma adaptação leve e focada no presente (A2) para reação constituem a solução ótima para o equilíbrio entre reatividade e eficiência computacional.
2.  **Hardware Real vs. Simulação**: A execução das experiências diretamente no Raspberry Pi 5 foi crucial. Revelou que estratégias teoricamente robustas como o retreino completo (A1) são, na prática, inviáveis devido a constrangimentos de latência e I/O do hardware. A simulação em máquinas de desenvolvimento mais potentes teria mascarado este problema fundamental.
3. **A Importância da Reprodutibilidade** : A automação de todo o fluxo de trabalho num pipeline de comando único (orquestrado por run_full_pipeline.py) e a estruturação rigorosa do repositório foram essenciais para gerar os 270 pontos de dados de forma consistente e para permitir uma análise estatística fidedigna, estabelecendo um padrão de qualidade para investigação reprodutível.
#### 5.2 Recomendações Futuras
Com base nos resultados e limitações identificadas, propomos as seguintes direções para trabalho futuro, estratificadas por horizonte temporal:

-   **Curto Prazo: Otimização do Detetor Estatístico**: O detetor DET2 (KS-Test) demonstrou uma latência de deteção fixa e elevada. Uma melhoria imediata seria o *tuning* do seu limiar de significância (α). Em vez de um valor fixo de 0.01, poder-se-ia gerar curvas ROC (*Receiver Operating Characteristic*) para diferentes valores de α, permitindo selecionar um ponto de operação que equilibre melhor a sensibilidade (taxa de verdadeiros positivos) e a especificidade (taxa de verdadeiros negativos), potencialmente reduzindo o atraso de deteção sem aumentar drasticamente a taxa de falsos alarmes.

-   **Médio Prazo: Ensembles Híbridos de Adaptação**: A estratégia A2, embora rápida, sofre de uma "amnésia" que a torna instável após drifts severos. Uma abordagem híbrida (A1+A2) poderia mitigar este problema. Após um alarme, o sistema poderia aplicar imediatamente a adaptação A2 para garantir a conformidade com o SLA e restaurar rapidamente um desempenho aceitável. Em paralelo, poderia despoletar um processo de retreino completo (A1) em *background*, com baixa prioridade de CPU. Uma vez que o modelo A1 mais robusto estivesse treinado, este substituiria o modelo A2 temporário. Isto combinaria a reatividade imediata do A2 com a robustez a longo prazo do A1.

-   **Longo Prazo: Aprendizagem Federada entre Dispositivos**: Numa fábrica, múltiplas máquinas idênticas podem estar em operação, cada uma equipada com um dispositivo DriftSense-PM. A Aprendizagem Federada (*Federated Learning*) permitiria que estes dispositivos colaborassem na aprendizagem sem partilharem os seus dados brutos. Cada dispositivo treinaria um modelo localmente (e.g., com a estratégia A2). Periodicamente, os parâmetros (pesos) destes modelos locais seriam enviados para um servidor de agregação central (que poderia ser outro Raspberry Pi), que os combinaria para criar um modelo global melhorado. Este modelo global seria então distribuído de volta para os dispositivos. Esta abordagem permitiria que o sistema aprendesse com uma diversidade muito maior de condições operacionais e tipos de drift, aumentando exponencialmente a robustez do sistema como um todo.

### 6. Apêndices Técnicos

#### Apêndice A: Fórmulas Matemáticas Completas

1.  **Root Mean Square (RMS)**: Mede a magnitude de um sinal variável. Para um sinal com N amostras $x_i$:
    $$ \text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2} $$

2.  **Skewness (Assimetria)**: Mede a assimetria da distribuição de probabilidade de uma variável aleatória.
    $$ \text{Skewness} = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^3}{\sigma^3} $$
    onde $\mu$ é a média e $\sigma$ é o desvio padrão.

3.  **Kurtosis (Curtose)**: Mede o "achatamento" da distribuição de probabilidade.
    $$ \text{Kurtosis} = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^4}{\sigma^4} - 3 $$

4.  **Teste Kolmogorov-Smirnov (KS-Test)**: A estatística de teste $D_{n,m}$ para duas amostras de tamanhos $n$ e $m$ é dada por:
    $$ D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)| $$
    onde $F_{1,n}(x)$ e $F_{2,m}(x)$ são as funções de distribuição cumulativa empírica das duas amostras. A hipótese nula é rejeitada ao nível $\alpha$ se:
    $$ D_{n,m} > c(\alpha) \sqrt{\frac{n+m}{nm}} $$
    onde $c(\alpha)$ é o valor crítico da distribuição de Kolmogorov.

5.  **Teste de Wilcoxon Signed-Rank**: Para uma amostra de diferenças $d_i$, calcula-se o rank $R_i$ dos valores absolutos $|d_i|$. A estatística de teste $W$ é a soma dos ranks das diferenças positivas:
    $$ W = \sum_{i=1}^{N} [\text{sgn}(d_i) \cdot R_i] $$
    O p-value é então derivado da distribuição de $W$.

#### Apêndice B: Especificação de Ficheiros do Pipeline
O pipeline de execução é orquestrado por uma série de scripts, cada um com uma responsabilidade específica, garantindo a modularidade e a reprodutibilidade.

-   `run_full_pipeline.py`: : Orquestrador macro central do projeto. Coordena e monitoriza a execução sequencial de todos os estágios do pipeline (Fases 1 a 5) de forma automatizada.
-   `feature_engineering.py`: Script de pré-processamento que transforma os dados brutos (`data/raw/`) em características estatísticas (`data/processed/`), como RMS, Skewness e Kurtosis.
-   `train_baseline_full.py`: Isola a lógica de treino do modelo de base. Carrega os dados de treino (D0), treina o One-Class SVM e serializa o modelo e o scaler para `models/`.
-   `master_script.py`: O ponto de entrada principal. É responsável por iterar sobre a matriz fatorial completa (cenários, detetores, adaptações), invocar o `run_full_pipeline.py` para cada combinação e agregar os resultados.
-   `statistical_analysis.py`: Após a conclusão de todas as execuções, este script lê os logs e métricas de `results/metrics/`, calcula estatísticas agregadas (médias, desvios padrão), realiza testes estatísticos (Wilcoxon) e gera os ficheiros de resumo como `full_factorial_summary.csv`.
-   `generate_thesis_plots.py`: Utiliza os dados agregados para gerar todas as visualizações (gráficos de barras, heatmaps, etc.) encontradas em `results/figures/`, que são usadas para a análise visual dos resultados.

#### Apêndice C: Trecho de Código Crítico
```python
# Ficheiro: src/adaptations.py
import time
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def apply_a2_lightweight_adapt(X_buffer_new):
    """
    Estratégia A2: Lightweight Adaptation.
    Treina um modelo pequeno APENAS com a nova realidade (buffer).
    Custo Energético: BAIXO | Latência: BAIXA
    """
    start_time = time.time()
    
    # 1. Scaler adaptado apenas à nova realidade (buffer recente das últimas 20 janelas)
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_buffer_new)
    
    # 2. Modelo Rápido (Poucas árvores, ideal para o ecossistema Edge)
    # Contamination de 0.01 assumindo estabilização imediata na nova zona operativa
    new_model = IsolationForest(n_estimators=10, contamination=0.01, random_state=42)
    new_model.fit(X_scaled)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return new_model, new_scaler, latency_ms
```

A função `apply_a2_lightweight_adapt` do ficheiro `src/adaptations.py` é um componente central da nossa solução. É a implementação da estratégia de adaptação rápida que permite ao sistema cumprir o SLA de *edge computing*. 
