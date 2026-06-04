Universidade do Minho
Escola de Engenharia
junho de 2026
DriftSense-PM: Concept Drift Detection and Adaptative
Predictive Maintenance at the Edge
Eduarda Pereira Gonçalo Ferreira Gonçalo Magalhães
PG61516 PG61525 PG

Mestrado em Engenharia Informática 2025/

Projeto em Engenharia Internet

3 de junho de 2026

Resumo
O presente trabalho enquadra-se na área da Manutenção Preditiva e deEdge Computingno contexto da Indústria
4.0. O projeto tem como objetivo conceber e planear uma solução para automatizar a deteção de anomalias e
deconcept driftem equipamentos industriais, utilizando como protótipo umaSmart Fan. Este desafio surge da
necessidade de tornar os modelos deMachine Learningmais resilientes e adaptáveis, reduzindo a dependência
de modelos estáticos que perdem precisão perante o desgaste mecânico e as alterações ambientais. A solução
proposta baseia-se na integração de dispositivos na periferia da rede, nomeadamente oArduino Pro Smart Industry
Kitpara a extração contínua de dados sensoriais e oRaspberry Pi 5para a orquestração, deteção dedrifte eventual
retreino dos modelos.

Foi definida uma arquitetura de aquisição e processamento local, com cenários controlados de injeção de falhas
e algoritmos de adaptação para a atualização autónoma do sistema. A solução será avaliada com base na eficácia
preditiva, latência e consumo energético, procurando identificar ostrade-offsentre precisão e custo computacional.
O trabalho preparatório e as práticas de reprodutibilidade adotadas visam demonstrar o potencial desta abordagem
para criar sistemas industriais mais autónomos, fiáveis e escaláveis. Os resultados experimentais comprovaram
que a estratégia de adaptação porFine-Tuning(Estratégia A2) se destaca como a abordagem mais viável para o
ecossistemaEdge, gerando umSpeedupde 19.3x face ao retreino global tradicional, com um custo energético
e computacional negligenciável. Adicionalmente, identificou-se o impacto severo doCatastrophic Forgettingna
manutenção do histórico saudável do modelo, sendo propostas linhas de investigação futura baseadas emReplay
Bufferspara a sua mitigação.

i
Abstract
This work falls within the field of Predictive Maintenance and Edge Computing in the context of Industry 4.0. The
project aims to design and plan a solution to automate the detection of anomalies and concept drift in industrial
equipment, using a Smart Fan as a prototype. This challenge arises from the need to make Machine Learning
models more resilient and adaptable, reducing reliance on static models that lose accuracy due to mechanical wear
and environmental changes. The proposed solution is based on integrating edge devices, namely the Arduino Pro
Smart Industry Kit for continuous sensor data extraction and the Raspberry Pi 5 for orchestration, drift detection,
and eventual model retraining.

A local acquisition and processing architecture was defined, with controlled fault injection scenarios and
adaptation algorithms for autonomous system updates. The solution will be evaluated based on predictive
effectiveness, latency, and energy consumption, aiming to identify the trade-offs between accuracy and
computational cost. The preparatory work and reproducibility practices adopted aim to demonstrate the potential
of this approach to create more autonomous, reliable, and scalable industrial systems. The experimental results
proved that the Fine-Tuning strategy (A2) stands out as the most viable approach for the Edge ecosystem,
generating a 19.3x speedup compared to traditional global retraining, with negligible computational and energy
costs. Furthermore, the severe impact of Catastrophic Forgetting on maintaining the model’s healthy history was
identified, prompting future research directions based on Replay Buffers for its mitigation.

ii
Índice
1 Introdução
1.1 Enquadramento e Motivação
1.2 Objetivos do Projeto.
1.3 Plano de Trabalhos Original e Desempenho
1.3.1 Cronograma e Fases de Execução
1.3.2 Alocação Estratégica de Recursos.
1.3.3 Matriz de Riscos e Estratégias de Mitigação
1.4 Levantamento de Requisitos da Solução.
1.4.1 Requisitos Funcionais
1.4.2 Requisitos Não Funcionais
2 Estado da Arte
2.1 Aquisição, Fusão de Dados e Engenharia de Características
2.2 Diagnóstico de Falhas Múltiplas e Previsão de Vida Útil.
2.3 Mitigação e Adaptação ao Concept Drift
2.4 Confiança e Operacionalidade – Explicabilidade e Fiabilidade (XAI Reliability).
2.5 Síntese das Lacunas e Posicionamento
3 Método
3.1 Arquitetura e Modelo do Sistema
3.2 Tecnologias e Decisões de Design.
3.3 Metodologia Experimental e Cenários de Drift
3.3.1 Cenário Base
3.3.2 Cenários de Injeção de Drift
3.4 Pipeline de Dados, Modelação e Orquestração.
3.4.1 Recolha e Feature Engineering
3.4.2 Avaliação e Seleção do Modelo Preditivo
3.4.3 Implementação e Orquestração do Sistema
3.4.4 Reprodutibilidade e Determinismo Computacional.
3.4.5 Contextualização Industrial e Viabilidade Operacional
4 Avaliação Experimental e Discussão de Resultados
4.1 Análise de Variabilidade Estatística e Dispersão
4.2 Comportamento Temporal e Características Latentes.
4.3 Desempenho dos Mecanismos de Deteção
4.3.1 Validação Estatística do Atraso de Deteção
4.4 Eficácia das Lógicas de Adaptação e Tempo de Recuperação.
4.5 Impacto Energético e Latência Computacional na Periferia
4.5.1 Validação Estatística de Significância Computacional
5 Conclusão e Trabalho Futuro
5.1 Síntese das Contribuições e Conclusões Principais.
5.2 Limitações Identificadas e Ameaças à Validade
5.3 Linhas de Investigação Futura.
Bibliografia
Anexo I - Proposta de Projeto
A Apêndice A
A.1 Injeção de Ruído Sintético (Cenário D3)
Calendarização das tarefas propostas Índice de Figuras
Diagrama da arquitetura distribuída do sistema na periferia da rede.
Montagem física de controlo (D0) evidenciando o acoplamento do nó sensorial à carcaça do motor.
cinemática.. Montagem experimental do cenário de controlo (D0), ilustrando o alinhamento do sensor sobre a base
Emulação deCovariate Drift(D1)
Equipamento submetido aoOperational Drift(D2)
Cenário composto D4, sobrepondo variabilidade térmica sazonal ao aumento agudo de regime rotacional.
interna.. Emulação de falhas transversais severas no cenário D5, acoplando desvio estrutural e corrupção sensorial
noIsolation Foreste noLocal Outlier Factorem contraste com o rigor preditivo do OC-SVM.. Matrizes de confusão para os três algoritmos avaliados, evidenciando o elevado número de falsos positivos
ocorrência de drift. Distribuição da variância do sinal (AccXStd) por cenário, evidenciando o aumento da dispersão perante a
Série temporal evidenciando a quebra abrupta na energia do sinal (AccXRMS) (D0 para D2).
Variação da frequência de pico fundamental perante diferentes condições operacionais..
Análise PCA ilustrando a separação e o desvio dos cenários de drift no espaço de características latentes..
Atraso de deteção discriminado por cenário de anomalia.
Matrizes de confusão ilustrando a ocorrência deCatastrophic Forgetting.
Evolução temporal do F1-Score evidenciando o colapso nas lógicas adaptativas.
Tempo de recuperação (Recovery Time) em janelas amostrais.
Consumo energético cumulativo comparando A0, A1 e A2.
Análise comparativa do custo temporal e de convergência entre as estratégias.
Duração prevista vs duração efetiva do trabalho realizado Índice de Tabelas
Distribuição de responsabilidades pelas tarefas do projeto
Riscos e Estratégias de Mitigação
Levantamento de Requisitos Funcionais
Levantamento de Requisitos Não Funcionais.
a periferia. Comparação do desempenho preditivo dos algoritmos de aprendizagem não supervisionada avaliados para
Comparação de Desempenho dos Algoritmos de Deteção deDrift.
Análise de significância estatística (Teste de Wilcoxon) para o atraso de deteção.
Degradação do desempenho preditivo (F1-Score) do modelo estático base (A0)..
Avaliação Global do F1-Score Preditivo entre estratégias.
Avaliação do custo computacional e ganho de desempenho (Speedup) das estratégias naEdge..
vii

Acrónimos
ACM Association for Computing Machinery.

CPU Central Processing Unit.

DC Direct Current.

F1 F1-score.

FFT Fast Fourier Transform.

FPR False Positive Rate.

FR Full Retraining.

GPIO General Purpose Input/Output.

IL Incremental Learning.

IoT Internet of Things.

ISO International Organization for Standardization.

KL Kullback–Leibler.

KS Kolmogorov–Smirnov.

ML Machine Learning.

PCA Principal Component Analysis.

PM Manutenção Preditiva.

PSI Population Stability Index.

RPLS Recursive Partial Least Squares.

RUL Remaining Useful Life.

SPC Statistical Process Control.

USB Universal Serial Bus.

YAML YAML Ain’t Markup Language.

viii
1 Introdução
O presente relatório documenta o trabalho e o planeamento detalhado desenvolvidos ao longo do segundo
semestre do ano letivo de 2025/2026, no âmbito da unidade curricular de Projeto em Engenharia Internet,
inserida no plano de estudos do Mestrado em Engenharia Informática da Universidade do Minho. O projeto
desenvolvido, intituladoDriftSense-PM: Concept Drift Detection and Adaptive Predictive Maintenance at the Edge,
teve início em fevereiro de 2026 e decorreu até ao final do mês de maio. O tema foi proposto pelo Professor Flávio
de Oliveira Silva, sendo o desenvolvimento acompanhado e validado pela equipa docente da unidade curricular,
coordenada pelo Professor Ivo Silva e acompanhada pelo Professor Bruno Antunes.

1.1 Enquadramento e Motivação
A crescente adoção de tecnologias associadas à Indústria 4.0 e a proliferação de dispositivos daInternet of Things
(IoT) impõem desafios significativos à gestão e monitorização de equipamentos industriais. Para responder a estas
exigências e minimizar os tempos dedowntime, torna-se necessário adotar processos cada vez mais proativos e
eficientes. Nesse contexto, a Manutenção Preditiva (PM) emerge como uma componente essencial, baseando-se
na recolha contínua de dados sensoriais visando a deteção antecipada de falhas e uma resposta rápida e segura às
degradações operacionais.

A eficácia dos sistemas de manutenção preditiva depende fortemente da precisão dos modelos deMachine
Learning(ML) implementados. Contudo, a maioria das soluções tradicionais assume, de forma limitadora, que a
distribuição dos dados é estacionária, o que raramente se verifica em ambientes físicos e industriais. Fatores
dinâmicos como a alteração de regimes de operação, o desgaste mecânico dos componentes, as flutuações
ambientais e o desgaste dos sensores introduzemconcept drift. Este fenómeno degrada significativamente a
fiabilidade dos modelos ao longo do tempo, exigindo abordagens que não dependam exclusivamente de
configurações estáticas e que se adaptem autonomamente.

A evolução do Edge Computing, aliada ao desenvolvimento de técnicas de aprendizagem automática
adaptativas, tem impulsionado a resiliência e a autonomia destas infraestruturas. A adoção destas soluções junto
à fonte de dados, permite reduzir a latência de inferência, contornar a dependência contínua de serviçosCloude
otimizar os recursos de rede. Neste panorama, destaca-se a importância de metodologias que promovam a
padronização experimental, a gestão eficiente do consumo energético, características essenciais para gerir os
trade-offsde ambientes industriais de elevado desempenho.

1.2 Objetivos do Projeto.
O presente projeto tem como objetivo principal conceber, implementar e avaliar experimentalmente umapipeline
de PM adaptativa e capaz de identificarconcept drift, operando integralmente na periferia da rede. Para simular um
ambiente industrial realista de monitorização, a solução integrahardwarede modo a reproduzir o comportamento
de uma máquina industrial e encontra-se estruturada numa arquitetura de três camadas. A primeira camada
(Physical/Sensing Node) é constituída por um protótipo de umaSmart Fan, acoplada a um ecossistema de aquisição
de alta frequência suportado pelo microcontrolador Arduino Portenta C33 e o sensor Nicla Sense ME, comunicando
viastreamde fluxo de dados série. A segunda camada (Edge Computing Node), composta por um Raspberry Pi 5,
assume a orquestração do sistema. Este, encarrega-se da extração temporal de dados, inferência do modelo de ML e
deteção dedrift, realizando deste modo uma monitorização de desempenho e divergência estatística e consequente
lógica de adaptação (retreino periódico oufine-tuning). Por fim, a terceira camada foca-se na monitorização e
avaliação rigorosa, garantindo o registo de métricas estruturadas e a análise do perfil de consumo energético do
sistema. Noseuconjunto, esteambientevisacomprovaraviabilidadedesistemasautónomoseresilientes, validando
ostrade-offsentre precisão preditiva, latência operacional e custo energético.

A proposta de projeto, presente noAnexo Icontempla as principais tarefas a serem realizadas, as quais são
listadas de forma mais desenvolvida abaixo, com o objetivo de orientar a execução do projeto de forma estruturada,
coerente e cientificamente reprodutível. Os principais objetivos considerados para alcançar as metas do projeto são
os seguintes:

Desenhar e implementar umpipelinede extração de dados em tempo real, utilizando sensores industriais
acoplados a equipamentos físicos na periferia da rede.
Desenvolver um ambiente controlado que possibilite a injeção física e virtual de falhas, como variações de
temperatura, alterações de montagem mecânica ou degradação de sensores, garantindo a validação das
soluções em condições replicáveis.
Implementar e comparar múltiplos mecanismos de deteção deconcept drift, contrastando abordagens de
monitorização de erro preditivo com testes de distribuição estatística.
InvestigareavaliarestratégiasdeadaptaçãodemodelosadequadasparaaexecuçãonaEdge, nomeadamente
comparando abordagens de retreino periódico com estratégias define-tuning.
Quantificar de forma rigorosa o impacto da perceção dedriftnas métricas de desempenho do sistema,
medindo explicitamente a precisão (F1-score), o atraso na deteção, a latência de inferência e o custo
energético por decisão.
Documentar, de forma detalhada, todo o processo, os protocolos experimentais adotados e estruturar um
repositório com código,datasetse configurações, visando a implementação de soluções semelhantes no
futuro.
1.3 Plano de Trabalhos Original e Desempenho
Este capítulo apresenta o planeamento da execução, a utilização dos recursos dehardwaree a matriz de gestão
de riscos, garantindo o cumprimento dos objetivos no prazo estipulado.

O trabalho foi dividido nas seguintes tarefas principais:
1.3.1 Cronograma e Fases de Execução
T1 – Planeamento e Definição de Âmbito: Realizar o planeamento técnico detalhado, arquitetura experimental
e documentar a estrutura de avaliação da unidade curricular.
T2 – Estado da Arte e Enquadramento Técnico-Científico: Analisar a literatura científica e soluções existentes
focadas em Manutenção Preditiva naEdgee nos mecanismos de deteção deconcept driftna Indústria 4.0.
T3 – Levantamento de Requisitos e Arquitetura da Solução: Especificar a taxonomia controlada de desvios,
definir as métricas de avaliação e documentar o protocolo experimental.
T4 – Aquisição de Dados e Pipeline Base: Montar a infraestrutura dehardware, integrando oArduino Pro
Kite oRaspberry Pi 5à estrutura da Smart Fan. Validar a calibração dos sensores, recolher odatasetde
controlo e finalizar treino do modelo no Cenáriobaseline.
T5 – Injeção deDrifte Algoritmos de Deteção: Injetar fisicamente as anomalias controladas no motor de
modo a construir as curvas de degradação. Implementar em paralelo os algoritmos de deteção baseados
em erro de performance e em distribuição estatística.
T6 – Estratégias de Adaptação e Avaliação: Desenvolver e pôr em prática mecanismos autónomos de
recuperação do modelo, incluindo o re-treino periódico de elevado custo e a adaptação porfine-tuning.
T7 – Análise de Resultados e Replicação: Executar autonomamente a matriz experimental completa. Realizar
a análise de significância estatística rigorosa (testes deWilcoxon) e estruturar o repositório de acordo com as
normas de reprodutibilidadeACM Artifact Evaluation.
T8 – Escrita do Artigo e Revisão Final: Consolidar conclusões sobre ostrade-offsanalisados, redigir o
documento científico final e preparar a demonstração da solução para a defesa de avaliação.
As tarefas acima descritas encontram-se apresentadas no cronograma ilustrado na Figura 1.
Figure 1: Calendarização das tarefas propostas
Semanas de projeto
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
T
T
T
T
T
T
T
T
A Tabela1.1apresenta uma comparação detalhada entre a duração prevista inicialmente para as tarefas do
projeto e a duração efetivamente despendida em cada uma das fases. Tal como esperado, nem sempre foi possível
concluir cada uma das tarefas no tempo estipulado inicialmente. Na T3 houve alguma dificuldade em conseguir
definir as métricas de avaliação e desenvolver uma pipeline experimental completa, na T4 tornou-se bastante morosa
a parte da calibração dos sensores e recolha do dataset de controlo, devido a problemas nohardware, e por fim,
na T7 onde foi realizda a análise de significância estatística, mais concretamente, os testes de Wilcoxon.

Table 1.1 Duração prevista vs duração efetiva do trabalho realizado
Tarefa Previsto (horas) Início (semana) Fim (semana) Efetivo (horas)
T1 5 1 2 5
T2 7 2 3 5
T3 8 3 5 11
T4 10 4 7 15
T5 15 5 7 13
T6 10 6 8 11
T7 11 7 8 13
T8 9 8 9 7
1.3.2 Alocação Estratégica de Recursos.
De modo a sustentar a arquitetura proposta e garantir o cumprimento dos objetivos delineados, os recursos do
projeto foram dimensionados com base no paradigma deEdge Computinge nas exigências de reprodutibilidade
científica. A alocação foi estruturada em três vetores fundamentais:

Disponibilidade de Hardware – O projeto conta com acesso a um nó de processamento Edge (Raspberry Pi
e um kit de sensorização industrial (Arduino Pro Smart Industry Predictive Maintenance Kit), garantindo a
fidelidade física necessária para a monitorização da Smart Fan.
Gestão de Ambiente de Software – Para mitigar riscos de incompatibilidade e assegurar a paridade entre os
postos de trabalho dos elementos da equipa, adotou-se o isolamento por contentorização (Docker) e a gestão
de dependências via Configuration as Code.
Divisão de Responsabilidades – Para efeitos de coordenação interna do projeto, foi atribuído a cada fase um
responsável principal e mantendo os restantes elementos como apoio técnico, de validação e de revisão.
Table 1.2 Distribuição de responsabilidades pelas tarefas do projeto
Tarefa Responsável principal Apoio
T1 Gonçalo Ferreira Eduarda Pereira, Gonçalo Magalhães
T2 Eduarda Pereira Gonçalo Ferreira, Gonçalo Magalhães
T3 Gonçalo Magalhães Gonçalo Ferreira, Eduarda Pereira
T4 Gonçalo Ferreira Gonçalo Magalhães, Eduarda Pereira
T5 Eduarda Pereira Gonçalo Magalhães, Gonçalo Ferreira
T6 Gonçalo Magalhães Eduarda Pereira, Gonçalo Ferreira
T7 Eduarda Pereira Gonçalo Ferreira, Gonçalo Magalhães
T8 Gonçalo Ferreira Eduarda Pereira, Gonçalo Magalhães
1.3.3 Matriz de Riscos e Estratégias de Mitigação
Na Tabela1.3estão presentes os riscos identificados para este projeto, juntamente com a probabilidade de
ocorrência, impacto, seriedade, os seus impactos/efeitos e a sua ação de mitigação. A cada um dos itens, para a
probabilidade e o impacto, é atribuída uma pontuação numa escala de 1 a 5, em que o 1 corresponde a baixo e
5 corresponde a alto. A seriedade de cada risco obtém-se multiplicando a probabilidade pelo impacto, permitindo
enaltecer os riscos que mais impacto poderão causar no projeto caso ocorram, de forma a estarmos mais atentos a
eles. A identificação dos riscos inerentes a estudos desta natureza permite que os mesmos possam ser prevenidos,
de forma a provocarem o menor dano possível ao longo do decorrer do projeto.

Table 1.3 Riscos e Estratégias de Mitigação
ID Risco Mitigação P I S
R1 Incumprimento dos prazos de entrega
Esforço suplementar para o cumprimento e
melhoria do planeamento
2 5 10
R
Impossibilidade de atingir os resultados
esperados
Rever o problema com o orientador e
adequar os objetivos
2 4 8
R
Falhas técnicas que comprometam o
projeto
Utilização de repositórios para controlo de
versões e realização de backups
2 4 8
R
Especificação incorreta dos requisitos e
objetivos do projeto
Esclarecer e discutir os requisitos com o
orientador
2 4 8
R
Impossibilidade de obtenção de meios para
a implementação do protótipo
Procurar meios passíveis de serem
implementados de forma mais acessível
2 3 6
R6 Quebra de Reprodutibilidade
Automatizar o registo de todas as
experiências para garantir que possam ser
repetidas com exatidão.
3 4 12
R7 Falta de Qualidade e Equilíbrio de Dados
Aplicar verificações automáticas na recolha
para garantir que o conjunto de dados seja
equilibrado e fiável
3 4 12
R8 Variação nos Testes Físicos
Delinear de forma rigorosa a montagem e
cenários de simulação dedriftpara que
todos os testes sejam idênticos.
4 4 16
1.4 Levantamento de Requisitos da Solução.
Este capítulo detalha a especificação técnica do projeto, fundamentada nos quatro objetivos centrais
estabelecidos. O levantamento de requisitos aqui apresentado visa não só assegurar a viabilidade operacional do
sistema, mas também garantir o cumprimento rigoroso dos critérios de desempenho e reprodutibilidade exigidos
para a validação científica do projeto.

1.4.1 Requisitos Funcionais
Na Tabela1.4apresentam-se os requisitos funcionais identificados e a sua relação com os objetivos definidos.
Estes requisitos têm o objetivo de assegurar que todas as funcionalidades necessárias sejam implementadas de

forma estruturada e eficiente.

Table 1.4 Levantamento de Requisitos Funcionais
ID Requisito Descrição Técnica Justificação /Fonte SecçãodeValidação
RF01 Aquisição de Dados naEdge
Implementação de pipeline com sensores
Arduino Pro Smart Industry Kite transmissão
para Raspberry Pi 5.
Objetivo 1 Secção4.
RF02 Injeção deDrift
Testar cenários de controlo, desvio térmico,
desvio de montagem mecânica, alteração de
regime/carga, degradação de sensor/ruído e
desvio combinado.
Objetivo 2 Secções4.1e4.
RF03 Deteção de Anomalias
Execução do detetor de erro preditivo (F1-
score) e do detetor de distribuição estatística,
avaliando oCovariate Drift.
Objetivo 2 Secção4.
RF04 Adaptação Autónoma
Avaliação de estratégias de re-treino periódico
e adaptação incremental (Fine-tuning e
recalibração de características).
Objetivo 3 Secções4.4e4.
RF05 Métricas de Avaliação
Extração deF1-score, atraso de deteção, taxa
de falsos alarmes, tempo de recuperação,
latência de inferência e custo energético por
decisão.
Objetivo 4 Secções4.3,4.4e4.
RF06 Integridade de Dados
Garantia de 1000 janelas/estado e
desequilíbrio de classes < 10% (Validation
Gate).
Objetivo 1 Secção4.
1.4.2 Requisitos Não Funcionais
Na Tabela1.5apresentam-se os requisitos não funcionais, nomeadamente as restrições físicas dohardware
industrial e os critérios de validação científica subjacentes à norma ACM.

Table 1.5 Levantamento de Requisitos Não Funcionais
ID Requisito Descrição Técnica Justificação /Fonte SecçãodeValidação
RNF01 Estabilidade e Latência
O processamento e os testes estatísticos não
podem introduzir bloqueios indeterminísticos.
O percentil 95 da latência deve manter-se
estável num limiar de variação máxima de
± 10 % entre repetições.
Rigor Científico Secção4.
RNF02 Medição Energética
Impacto energético deve ser reportado de forma
explícita, recorrendo a instrumentação física
dedicada à medição contínua do consumo na
Edge.
Objetivo 4 Secção4.
RNF03 Reprodutibilidade
Toda a pipeline experimental deve ser
determinística. A execução deve ocorrer num
ambiente Docker, sendo orquestrada por
umscriptúnico de ponto de entrada com
documentação estruturada.
Norma ACM ValidadoTransversalmente
RNF04 Validação Estatística
O plano de avaliação deve executar
automaticamente a matriz de testes, com
5 repetições por configuração, reportando
intervalos de confiança de 95% e o teste
Wilcoxon signed-rank para comparar o
desempenho dos mechanismos de deteção.
Método Científico Secção4.3.
É importante destacar o RNF02, a medição energética rigorosa do impacto energético em dispositivos físicos
Edge, uma das contribuições mais originais e relevantes deste trabalho. Ao invés de avaliações puramente teóricas,
a quantificação física deste custo prova o verdadeiro valor e viabilidade desta arquitetura na Indústria 4.0.

2 Estado da Arte
A Manutenção Preditiva no contexto da Indústria 4.0 representa uma mudança de paradigma face às
abordagens reativas e preventivas, suportando-se na monitorização contínua de condição para antecipar falhas
[ 1 ]. Os avanços recentes, apoiam-se fortemente em técnicas de Inteligência Artificial e fusão de dados na área de
Internet of Things[ 2 ]. Contudo, a viabilidade operacional destes sistemas é severamente ameaçada pela
presunção clássica da aprendizagem automática de que os dados de treino e teste partilham uma distribuição
estatística idêntica e independente [ 3 ]. Em ambientes industriais reais, a degradação dos componentes, a
substituição de sensores, as variações sazonais e as flutuações nas cargas de trabalho induzem o fenómeno de
Concept Drift. Modelos treinados de forma estática sofrem uma degradação de desempenho, nomeadamente na
sensibilidaderecall, quando expostos a estes dados não estacionários, o que resulta em falsos alarmes ou, mais
criticamente, na falha de deteção de anomalias. O presente estado da arte sistematiza a literatura em torno da
aquisição de dados, modelação preditiva, estratégias de mitigação dedrift, explicabilidade e a sua transição para a
Edge Computing[ 4 ].

2.1 Aquisição, Fusão de Dados e Engenharia de Características
A qualidade e a representação dos dados são os alicerces da Manutenção Preditiva, a literatura recente
evidencia que abordagens suportadas apenas no domínio do tempo podem ser insuficientes devido à
complexidade do ruído mecânico e elétrico. A deteção precoce de falhas exige a integração de assinaturas de
degradação distintas. Em [ 2 ] demonstraram a superioridade da fusão multi-sensorial (vibração, temperatura e
corrente elétrica) utilizando dados brutos baseados na Transformada deFourier, contornando as limitações da
extração tradicional no domínio do tempo. Adicionalmente, em sistemas de transmissão complexos, ferramentas
como a análise de ordem ciclo estacionária, que estabiliza características sob condições de velocidade e carga
variáveis provaram ser essenciais para isolar componentes moduladas ligadas ao desgaste das engrenagens e
rolamentos, mitigando os efeitos do ruído dos inversores elétricos [ 5 ]. Uma vulnerabilidade frequentemente
ignorada no treino de algoritmos de PM é o viés estático. Modelos baseados exclusivamente em variáveis
acumuladas, como horas de funcionamento, temperatura média, tendem a memorizar a correlação temporal da
máquina em vez da verdadeira assinatura de falha mecânica. [ 4 ] demonstrou que a incorporação de métricas
dinâmicas, como os coeficientes derivativos Slope e Delta, atuam como um filtro. Esta engenharia de
características dinâmicas isola o processo de mudança contínuo, tornando o modelo imune a variações sazonais
lentas, neutralizando assim falhas sistémicas no desempenho durante avaliações cronológicas estritas.

2.2 Diagnóstico de Falhas Múltiplas e Previsão de Vida Útil.
A literatura tem evoluído do diagnóstico de falhas isoladas, assumindo a independência dos componentes, para
a modelação de falhas simultâneas e quantificação da incerteza. A deteção simultânea e otimização baseada em
custo, onde [ 6 ] identifica como lacuna significativa a incapacidade dos modelos tradicionais gerirem múltiplas falhas
de forma simultânea em ambientes evolutivos. A abordagem deMulti-Label Classification(MLC) é crucial, uma vez
que a degradação de um componente frequentemente precipita a falha de sistemas adjacentes. De uma perspetiva
financeira, [ 1 ] combinou técnicas deMachine Learningcom o métodoBest-Worst(BWM) para priorizar falhas. O
uso de métricas focadas não apenas na precisão teórica, mas no impacto económico das falsas deteções e falhas
críticas ignoradas é imperativo para a aceitação na indústria. A precisão na estimativa da Vida Útil Restante é
frequentemente prejudicada por degradações não-lineares. Para além disso, [ 1 ] aplica a integração de Modelos
Probabilísticos de Espaço de Estados Profundos com inferência variada para quantificar incertezas. A otimização
heurística, provou ser vital para ajustar de forma eficiente redes neuronais altamente dimensionais, com métricas
de validação como oPrediction Interval Coverage Probability(PICP) a oferecerem garantias probabilísticas críticas
para o agendamento de intervenções, em oposição a simples previsões determinísticas [ 7 ].

2.3 Mitigação e Adaptação ao Concept Drift
A taxonomia doConcept Driftclassifica as mudanças da distribuição de dados como súbitas, incrementais,
graduais ou recorrentes. O tratamento deste fenómeno é abordado na literatura em duas vertentes
complementares, deteção e adaptação ativa. Os métodos de base de deteção de desvio assentam em modelos
analíticos como oDrift Detection Method(DDM), que monitoriza flutuações na taxa de erro de classificação, e os
Window-Based Methods(WBM), que comparam as propriedades estatísticas de blocos de dados históricos face a
dados recentes. Estes métodos equilibram o rigor computacional com a precisão, onde a resposta aodriftexige
frequentemente re-treino. Para mitigar o dilema da estabilidade, onde aprender novos conceitos apaga
conhecimentos passados (catastrophic forgetting), aContinual Machine Learning(CML) emerge como solução
primária. [ 3 ] detalha uma arquitetura de referência capaz de detetar desvios e orquestrar atualizações
automáticas de modelos. Para cenários estritamente destreaming, algoritmos como as Árvores Adaptativas de
Hoeffdingfornecem adaptação de alta velocidade. [ 6 ] focou-se noOnline Ensemble of Multi-Label Hoeffding
Adaptive Trees(OEMLHAT). Ao utilizar reamostragem baseada numa distribuição dePoissonpara simularbagging
online, o modelo consegue processar dados infinitos sem os armazenar, ramificando ou substituindo sub-árvores
embackgroundsempre que o desvio estatístico ultrapassa o limite de confiança deHoeffding 4.3. Para além
disso, avaliações com base em validação cruzada aleatória, comoK-foldem dados não estacionários resultam em
estimativas artificialmente otimistas, com falhas na transferência para operação real. A validação cronológica, na
qual o modelo é estritamente testado em blocos temporais futuros não observados, deve ser o método principal

para quantificar a verdadeira resistência aoConcept Drift. [ 8 ] corrobora este facto, sublinhando que modelos
adaptativos que executam re-treino incremental mantêm taxas de resiliência e estabilidade significativamente
superiores em cenários de degradação prolongada face aos seus homólogos estáticos.

2.4 Confiança e Operacionalidade – Explicabilidade e Fiabilidade (XAI Reliability).
Relativamente aExplainable Artificial Intelligence(XAI), as técnicas agnósticas a modelos, como oLocal
Interpretable Model-Agnostic Explanations(LIME), e análises globais de importância em modelos deRandom
Forestfornecem interpretabilidade bidimensional [ 2 ]. O cruzamento das predições com XAI fomenta o ciclo
Socialization, Externalization, Combination, Internalization (SECI), onde o conhecimento empírico e tácito dos
operadores é fundamentado e externalizado pelas inferências explícitas matemáticas extraídas do algoritmo,
fortalecendo a adoção na linha de produção. [ 9 ] abordou a incerteza nos dados implementando um mecanismo
RIML de arquitetura bi-fásica. O modelo aprende a classificar a falha e simultaneamente estima os valores
intrínsecos de um conjunto de sensores críticos. Durante a fase de inferência, o desvioentre as leituras reais dos
sensores e as estimadas pelo modelo atua como um filtro (Pass/Discard), descartando predições com alto risco
dedata drifte aumentando robustamente a exatidão útil sobre os dados retidos. Apesar da sofisticação
metodológica em análise de dados contínuos, os ambientes de IoT industriais sofrem restrições severas de largura
de banda e latência, obrigando a que o processamento seja deslocado daCloudpara a borda aEdgeda rede.
Para que implementações de deteção de drift ocorram emhardware restrito, como osRaspberry Pi ou
microcontroladores, as soluções exigem compromissos entre a complexidade e odetection delaycontra o
consumo energético imperioso gerado na execução de arquiteturas de IA [ 7 , 9 ]. A literatura demonstra um
desenvolvimento teórico prolífico nas vertentes de IA e Big Data, mas é substancialmente carente em
implementações empíricas abrangentes naEdgeque afiram diretamente estas penalizações físicas.

2.5 Síntese das Lacunas e Posicionamento
Uma análise crítica às publicações citadas revela que, apesar da Manutenção Preditiva e oConcept Drift
atraírem uma atenção massiva isolada, persistem lacunas interdisciplinares graves, como por exemplo, a ausência
de avaliação holística de impacto naedge, embora modelos adaptativos incrementais e de XAI atinjam alta
precisão [ 1 , 6 ], falta literatura empírica e reprodutível que mapeie a sua exigência computacional emhardware
Edge, ponderando otrade-offlatência/exatidão e consumo de energia em regime adaptativo contínuo. Para além
disso, outra lacuna importante a referir é a deficiência em ambientes de validação controlada, onde a maioria das
abordagens recorre a avaliações em bases de dados genéricas, onde a natureza e otimingdo desvio não estão
documentados ou não são controlados. Avaliar algoritmos adaptativos requer cenários metodológicos com injeção

dedriftrigorosa e reproduzível, aplicando tempos concretos de re-adaptação algorítmica face à introdução forçada
de anomalias. A verdadeira implementação da Indústria 4.0 exige que os algoritmos abandonem os redutos de
bases de dados estáticas e abracem o processamento adaptativo e contínuo, resistente aos processos
degenerativos da operação real. O presente projeto posiciona-se precisamente na intersecção desta vanguarda,
endereça a urgência teórica da deteção robusta de desvios através de mecanismos de injeção controlada,
colmatando simultaneamente a lacuna operacional através da sua avaliação infraestrutural e medição de custos
energéticos diretamente na camadaEdge.

3 Método
Neste capítulo é detalhada toda a solução desenvolvida, incluindo a arquitetura, tecnologias, decisões técnicas e
diagramas explicativos.

3.1 Arquitetura e Modelo do Sistema
A topologia do sistema apresenta uma arquitetura distribuída integralmente executada na periferia da rede (Edge).
O modelo foi concebido para eliminar a dependência de orquestração emCloud, garantindo baixa latência de
inferência, privacidade estrutural dos dados e capacidade de adaptação no local perante cenários de variabilidade
temporal.
A arquitetura organiza-se em três camadas lógicas e físicas interdependentes: a Camada Físico-Sensorial, a
Camada de Processamento e Orquestração e a Camada de Observabilidade e Avaliação.

Figure 2: Diagrama da arquitetura distribuída do sistema na periferia da rede.
A Camada Físico-Sensorial é composta pela planta mecânica materializada num Motor DC (Smart Fan),
submetido a perturbações externas controladas (térmicas e mecânicas). Acoplado fisicamente à estrutura encontra-
se oArduino Pro Smart Industry Kit. O sensorNicla Sense MEprocede à amostragem de alta frequência dos eixos de
vibração, temperatura e humidade. Por sua vez, o Arduino atua como umgatewayde aquisição inicial, transmitindo
dados em contínuo para a camada superior.

A Camada de Processamento e Orquestração é constituída pelo Raspberry Pi 5, que atua como recetor
de dados e controlador dinâmico do motor. Opipelineinterno executa-se de forma sequencial e cíclica:

Feature Extraction – Conversão de dados brutos em janelas temporais e extração de características no
domínio do tempo (média, variância) e da frequência (FFT).
ML Inference Model – O núcleo preditivo base (baseline) assenta num modelo deOne-Class Support
Vector Machine(OC-SVM). Esta arquitetura foi selecionada após uma calibração cruzada, demonstrando
superioridade na delimitação da fronteira geométrica dos dados em regime saudável (D0). O modelo foi
instanciado com umkernel Radial Basis Function(RBF) para mapear eficazmente a não-linearidade das
vibrações mecânicas. O limite de anomalia foi fixado emν= 0. 05 , garantindo que o hiperplano de decisão
contemple uma margem estrita que limite os falsos positivos a um máximo teórico de 5% durante a calibração
com dados exclusivamente nominais.
Drift Detection Engine – A orquestração doConcept Driftefetua-se através de dois motores analíticos
concorrentes:
- DET1 (Monitorização de Desempenho Preditivo): O estado dedrifté formalmente declarado
se o OC-SVM sinalizar inferências anómalas (ypred=− 1 ) durante um período deP= 10janelas
temporais consecutivas. O valorP = 10foi determinado empiricamente para absorver picos
transientes mecânicos naturais do motor sem acionar falsos alarmes, estabilizando o sistema
perante ruído espúrio.
- DET2 (Análise de Divergência Estatística): Aplica o teste não paramétrico de
Kolmogorov-Smirnov. O sistema aciona o alarme dedriftse op-valuefor inferior aα= 0. 001. Este
limiar conservador foi escolhido deliberadamente para mitigar falsos positivos induzidos pela elevada
variância de alta frequência inerente ao ambiente industrial.
Adaptation Logic – Perante a sinalização de um desvio no espaço vetorial, o sistema orquestra a
recuperação através de duas políticas de reconstrução do espaço latente:
- Estratégia A1 (Retreino Periódico Global): Atua como um processo determinístico e
independente do estado de alarme, sendo executado ciclicamente a cada 50 janelas de observação.
Esta intervenção força o recálculo total dos parâmetros do escalonador e a reconstrução da fronteira
do hiperplano do OC-SVM recorrendo à totalidade da memória alocada.
- Estratégia A2 (Fine-Tuning): Desenhada especificamente para mitigar a latência naEdge, opera
estritamente por interrupção (acionada pelo diagnóstico do DET1 ou DET2). O modelo aplica uma
atualização cirúrgica da fronteira de decisão utilizando apenas umbufferde memória recente, restrito
a 20 instâncias, minimizando o custo de re-convergência computacional.
Por fim, a Camada de Observabilidade e Avaliação dedica-se a garantir o rigor científico e a
reprodutibilidade das métricas (ACM Artifact Evaluation ready). Através de um medidor de energia físico (USB
Power Meter) colocado na fonte de alimentação do Raspberry Pi, capturam-se métricas termoelétricas contínuas.
Esses vetores de dados físicos são posteriormente submetidos a testes não paramétricos de significância
estatística.

3.2 Tecnologias e Decisões de Design.
A implementação desta solução preditiva impôs um conjunto de decisões dedesigncomplexas, onde cada
escolha tecnológica foi ponderada face aostrade-offscientíficos fundamentais entre precisão analítica, latência
e custo computacional em ambientes industriais restritos.
No domínio dohardware, a arquitetura tira partido da potência de cálculo do Raspberry Pi 5, orquestrador
central da lógica deMachine Learning, operando em simbiose com o microcontrolador Arduino Portenta C33 e o
sensor Nicla Sense ME. Esta dissociação entre aquisição sensorial e processamento lógico revela-se decisiva: ao
controlar o ciclo de trabalho (duty cycle) programaticamente, o sistema consegue automatizar a injeção de falhas e
alterar dinamicamente o regime motriz. Esta autonomia elimina a necessidade de intervenção mecânica humana
entre repetições experimentais, assegurando reprodutibilidade normativa irrefutável.
No que concerne ao ambiente aplicacional, a solução explora a versatilidade da linguagemPythone o rigor
matemático da biblioteca SciPy, otimizando o cálculo de divergências estatísticas como Kullback-Leibler e
Kolmogorov-Smirnov. Todo o ecossistema está encapsulado viaDocker, com fixação estrita de dependências,
garantindo portabilidade absoluta do nó deEdge.

3.3 Metodologia Experimental e Cenários de Drift
Para mitigar variações indeterminísticas e assegurar a reprodutibilidade metodológica, o comportamento da
plataforma mecânicaSmart Fanfoi condicionado a seis protocolos de ensaio padronizados. Esta abordagem
fatorial exaustiva isola as assinaturas sensoriais de cada fenómeno, gerando uma base empírica estrita para
avaliar a sensibilidade computacional da rede neuronal.

3.3.1 Cenário Base
O cenário de controlo inicial, designado por D0, estabelece o estado fundamental de referência operacional
(In-Control) da planta. A montagem mecânica do protótipo industrial e o acoplamento físico do ecossistema de
sensorização estão detalhados na Figura 3 e na Figura 4.
Neste protocolobaseline, o motor DC opera estabilizado a uma potência equivalente a 50% do seu ciclo de
trabalho nominal, durante um período contínuo de 1200 janelas temporais. O nóNicla Sense MEencontra-se

Figure 3: Montagem física de controlo (D0) evidenciando o acoplamento do nó sensorial à carcaça do motor.
fixado rigidamente à estrutura metálica vibratória de suporte.

Figure 4: Montagem experimental do cenário de controlo (D0), ilustrando o alinhamento do sensor sobre a base
cinemática.

A recolha de dados sob estas condições estacionárias parametriza as distribuições iniciais latentes, balizando
os limiares operatórios normais que quantificam as taxas de erro dos algoritmos de deteção subsequentes.
Na Figura 3 , na imagem da esquerda, é possível ver o motor DC de 3V equipado com uma hélice direcional,
utilizado para introduzir carga mecânica e gerar o comportamento físico a ser monitorizado. É também visível a
implementação física do circuito de mitigação de ruído eletromagnético. Este circuito é composto por três
condensadores cerâmicos de 10 nF, soldados diretamente nos terminais de alimentação e interligados à carcaça
metálica do motor, atuando simultaneamente como filtros de modo diferencial e de modo comum. Imediatamente
abaixo do motor, observa-se o núcleo de ferrite toroidal, em torno do qual os cabos de alimentação foram
enrolados e fixados, funcionando como um choke para bloquear a propagação conduzida de ruído de alta
frequência.

Na imagem da direita da mesma figura, destaca-se a montagem da placa de aquisição de dados, o Nicla Sense
ME, que se encontra firmemente acoplada à estrutura do motor através de abraçadeiras plásticas. Esta proximidade
física é fundamental para garantir a recolha precisa de métricas diretamente da fonte, como dados inerciais de
vibração, no entanto, tornava o microcontrolador altamente suscetível à forte interferência eletromagnética gerada
pelo movimento contínuo das escovas internas do motor DC. Sem a intervenção de hardware visível na imagem
da esquerda, a indução deste ruído provocava o bloqueio (freezing) imediato do Nicla Sense ME. A combinação da
filtragem capacitiva na origem do motor com a barreira indutiva na cablagem revelou-se indispensável para isolar
eletronicamente os dois sistemas, garantindo a estabilidade do microcontrolador e a recolha ininterrupta e fidedigna
dos dados experimentais.

3.3.2 Cenários de Injeção de Drift
De forma a quebrar o estado de equilíbrio, desenharam-se cinco protocolos distintos de stress dinâmico,
abordando desde flutuações contextuais ligeiras a desvios operacionais catastróficos compostos.

Emulação de Variação Ambiental (D1 - Covariate Drift Térmico): O cenário D1 modela uma alteração
isoladanadistribuiçãodascovariáveistérmicasdeentrada, semafetarodesgasteintrínsecomecânicodafronteirade
decisão. A injeção, ilustrada na Figura 5 , efetua-se introduzindo calor ambiente contínuo e progressivo na vizinhança
imediata dos transdutores. Esta dinâmica emula os ciclos térmicos de pavilhões fabris, testando a imunidade dos
detetores face à dilatação colateral nas leituras microeletromecânicas do acelerómetro.

Figure 5: Emulação deCovariate Drift(D1)
Alteração de Carga de Trabalho (D2 - Operational Drift por Regime): Devido à inexequibilidade de
garantir desapertos mecânicos precisos a ritmo constante laboratorial, emulou-se o desgaste estrutural por via de
sobrecarga. O protocolo D2 (Figura 6 ) impõe um salto programático no atuador PWM, catapultando o regime do

motor de 50% para 75% da capacidade máxima. Esta ação desloca de imediato a frequência fundamental
espectral, validando a velocidade de reação dos mecanismos de mitigação na resposta a quebras imediatas de
performance.

Figure 6: Equipamento submetido aoOperational Drift(D2)
Simulação de Degradação de Transdutores (D3 - Sensor Bias / Ruído): O cenário D3 materializa a
falha cumulativa e a corrupção silenciosa do hardware de aquisição. Para garantir reprodutibilidade total, o desvio
foi processado no domínio lógico (o algoritmo de injeção encontra-se detalhado no Apêndice A). O sinal degradado
foi gerado sinteticamente através da adição de ruído Gaussiano, formalizado pela equação:

XD 3 =XD 0 +N(0, σ) +bias (3.1)
ondeσrepresenta 15% do desvio padrão original dos dados de controlo (XD 0 ), e o termobiasintroduz uma
decalagem constante nos eixos vibratórios, emulando a perda de calibração eletrónica.

Cenário de Degradação Composta (D4 - Combined Drift Térmico e Operacional): O protocolo de
stresse D4 (Figura 7 ) eleva a complexidade fundindo vetores de anomalia. Justapõe a degradação gradual
induzida peloCovariate Drifttérmico (D1) à mudança estrita de rotação e carga imposta peloOperational Drift
(D2). A avaliação simultânea avalia se os detetores isolam causas múltiplas num ambiente multi-variável instável.

Figure 7: Cenário composto D4, sobrepondo variabilidade térmica sazonal ao aumento agudo de regime rotacional.

Degradação Composta de Larga Escala (D5 - Combined Drift Operacional e Estocástico): Sendo a
condição laboratorial mais hostil e próxima do estado de fim de vida útil de uma máquina industrial, o cenário D5
(Figura 8 ) ataca a rede neural preditiva em duas frentes vitais. Cruza o erro de conceito físico originado pela nova
assinatura vibratória de alta potência (D2) com a quebra catastrófica de integridade de dados resultante do ruído e
perda de calibração eletrónica do transdutor (D3).

Figure 8: Emulação de falhas transversais severas no cenário D5, acoplando desvio estrutural e corrupção sensorial
interna.

3.4 Pipeline de Dados, Modelação e Orquestração.
Após a criação de cenários dedrift, procedeu-se à recolha dos dados, tratamento dos mesmos, avaliação e
seleção de um modelo preditivo e implementação e orquestração do sistema.

3.4.1 Recolha e Feature Engineering
Abasemetodológicadestesistemaassentana extraçãorigorosadedados sensoriaisnaperiferiada rede. Durante
a fase de operação normal (Cenário D0), os dados brutos de vibração (AccX, AccY, AccZ) e temperatura foram
recolhidos em contínuo através da comunicação entre o Arduino Portenta e o Raspberry Pi 5. De modo a transformar
esta série temporal bruta em informação útil para os algoritmos deMachine Learning, implementou-se o módulo
feature_engineering.py. Estescriptrecorre à bibliotecaSciPypara processar as janelas de observação, extraindo
características vitais tanto no domínio do tempo (como a média, variância e RMS) como no domínio da frequência
(através da Transformada Rápida de Fourier - FFT).

3.4.2 Avaliação e Seleção do Modelo Preditivo
Após a padronização dos dados, tornou-se imperativo selecionar o algoritmo de deteção de anomalias mais
adequado às restrições daEdge. Para tal, através doscript train_baseline_full.py, conduziu-se uma avaliação
comparativa rigorosa entre três arquiteturas de aprendizagem não supervisionada: Local Outlier Factor(LOF),
Isolation Forest(ISF) eOne-Class Support Vector Machine(OC-SVM).
A comparação feita na Tabela3.1foi suportada pela análise detalhada das respetivas Matrizes de Confusão
(ilustradas na Figura 9 ) e da métricaF1-Score. Enquanto o LOF apresentou dificuldades na estabilização de
densidades locais variadas e oIsolation Forestrevelou uma delimitação de fronteiras demasiado permissiva
(resultando num elevado número de Falsos Positivos, visível na matriz de confusão correspondente), o modelo
OC-SVM destacou-se categoricamente.

(a) Isolation Forest (b) Local Outlier Factor (c) One-Class SVM
Figure 9: Matrizes de confusão para os três algoritmos avaliados, evidenciando o elevado número de falsos positivos
noIsolation Foreste noLocal Outlier Factorem contraste com o rigor preditivo do OC-SVM.

Configurado com umkernelRBF (Radial Basis Function) e um hiperparâmetro de saturação de anomaliasν=

05 , o OC-SVM demonstrou uma superioridade notável na definição geométrica do regime saudável da máquina.
Face a estes resultados, este foi o modelo selecionado para avançar para a fase de testes dinâmicos. A otimização
fina de hiperparâmetros (como a persistência e oAlphado teste KS) foi isolada numscript(optimize_detectors.py),
mantendo opipelineprincipal determinístico.
Table 3.1 Comparação do desempenho preditivo dos algoritmos de aprendizagem não supervisionada avaliados
para a periferia.

Modelo Classe Anomalia/Drift (-1) Accuracy Weighted Avg
Precision Recall F1-Score F1-Score Support
Isolation Forest (ISF) 1.000 0.099 0.180 0.134 0.177 619
Local Outlier Factor (LOF) 0.984 0.625 0.765 0.630 0.740 619
One-Class SVM (OC-SVM) 0.974 0.882 0.926 0.864 0.897 619
A seleção restrita a estes três algoritmos, LOF, ISF e OC-SVM , resultou de uma decisão arquitetural
intrinsecamente ligada às limitações dehardwarecaracterísticas do paradigmaEdge Computing. Tendo em conta
que o sistema de inferência opera numRaspberry Pi, a adoção de modelos mais complexos baseados emDeep
Learning, comoAutoencodersou Redes Neuronais Profundas foi liminarmente descartada. Embora essas
arquiteturas profundas possuam uma elevada capacidade de extração de características, exigem um poder de
processamento massivo, frequentemente dependente de aceleração por GPU, memória RAM e um consumo
energético incomportável para a periferia da rede. Em contraste, os algoritmos clássicos deMachine Learning
selecionados caracterizam-se por possuírem uma reduzida memory footprint e uma baixa complexidade
computacional durante a fase de inferência. Estas propriedades garantem uma execução na ordem dos
milissegundos, cumprindo rigorosamente os requisitos de baixa latência, operação em tempo real e eficiência
energética impostos pelo nóEdge.

3.4.3 Implementação e Orquestração do Sistema
A operacionalização da arquitetura proposta assenta numa estratégia de orquestração modular e hierárquica,
desenhada explicitamente para garantir a reprodutibilidade das experiências e conformidade com os requisitos de
avaliação de artefatos da ACM.
A camada de orquestração superior é operacionalizada pelorun_full_pipeline.py, que funciona como
ponto de entrada único da experiência. Estescriptexecuta a sequência completa dapipeline, desde a aquisição
de dados até à consolidação final de métricas e visualizações, garantindo determinismo através de configurações
parametrizadas em YAML e fixação deseedsaleatórias.

No núcleo da simulação experimental encontra-se omaster_script.py, responsável por orquestrar
autonomamente a atriz fatorial 3 × 3 (Detetores × Estratégias de Adaptação) ao longo de 5 repetições
independentes para cada cenário de injeção de falhas. Estescriptimplementa o seguinte fluxo:

Iteração sobre cenários de drift (D0–D4) com carregamento automático dos dados processados;
Ciclo de treino e teste para cada configuração (detetor + estratégia de adaptação);
Inferência sequencial sobre o conjunto de teste, sincronizada com a amostragem de métricas;
Acionamento condicional de adaptações quando um conceito dedrifté detetado.
Quando um concept drift é assinalado pelo detetor ativo, o fluxo de execução chama o módulo
adaptations.py, que implementa as funções lógicas de mitigação: (A1) retreino total do classificador com o
histórico completo de dados saudáveis, ou (A2)fine-tuningincremental sobre um subconjunto reduzido. A escolha
da estratégia é parametrizável e propagada através de um dicionário de configuração global.
A observabilidade e medição é garantida por vetores de registo paralelos, o
power_meter_fnirsi_windows.py coleta telemetria energética em tempo real via interface com o
equipamento FNIRSI, enquantotimersde inferência são capturadosinlineno código de deteção. Todas as
métricas (F1-score,delayde deteção, latência, consumo) são registadas estruturadamente em formato CSV e
serializadas para pós-processamento.
A análise e consolidação é automatizada pela cadeia descriptsanalíticos:statistical_analysis.py
processa as matrizes experimentais e executa testes de significância estatística (testes emparelhados de Wilcoxon
com correção de Bonferroni), enquantogenerate_thesis_plots.pyrenderiza visualizações de alta fidelidade
(gráficos de série temporal, matrizes de confusão, análises PCA) com estilo tipográfico consistente com o presente
projeto.
Esta estrutura hierárquica de componentes, conjugado com a codificação de toda a configuração experimental
em ficheiros de dados (YAML + CSV), assegurou a total reprodutibilidade da campanha experimental e permitiu
gerar os resultados empíricos que são apresentados e analisados criticamente no capítulo seguinte.

3.4.4 Reprodutibilidade e Determinismo Computacional.
A reprodutibilidade rigorosa de experiências em aprendizagem automática exige não apenas a documentação de
algoritmos, mas a fixação explícita de todas as dependências, configurações e geradores de aleatoriedade. Este
projeto cumpre as normas de Avaliação de Artefatos da ACM através de um protocolo estruturado de determinismo
computacional.
Encapsulamento em Container Docker – Todo o ambiente experimental foi containerizado num ficheiro
Dockerfileque especifica a imagem base (python:3.11-slim-bullseye), instalação de bibliotecas do

sistema (libopenblas-dev,liblapack-devpara operações numéricas) e todas as dependências Python.
Este contentor garante a paridade ambiental entre distintos postos de trabalho, eliminando a variabilidade decorrente
de incompatibilidades de versões de sistema operativo ou compilador.
Congelamento de Dependências – Todas as bibliotecas Python foram explicitamente selecionadas com as
versões específicas num ficheirorequirements.txt:

scikit-learn==1.3.2(para OC-SVM,Isolation Foreste LOF);
numpy==1.24.3(operações matriciais determinísticas);
scipy==1.11.2(testes estatísticos Wilcoxon e KS);
pandas==2.0.3(manipulação dedatasets);
matplotlib==3.7.1eseaborn==0.12.2(renderização de gráficos).
Inicialização de Seeds Aleatórias – Todos osscriptsde experimentação fixam explicitamente asseedsde
números pseudoaleatórios no início da execução:
Listing 3.1: Fixação de seeds aleatórias para garantir determinismo.
import numpy as np
from sklearn.utils import check_random_state
import random

RANDOM_SEED = 42 # Fixo documentado
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

Esta fixação aplica-se a: (i) divisão treino/teste nosscriptsde recolha, (ii) inicialização de centroides em
clustering, (iii) amostragem estocástica emensemble methods, e (iv) permutações em testes de significância
estatística.
Configurações Parametrizadas – Todos os hiperparâmetros experimentais (e.g.,nu=0.05para OC-SVM,
P=10 para persistência de deteção, alpha=0.001 para teste KS) foram centralizados num ficheiro
config.yaml:

Listing 3.2: Exemplo da parametrização centralizada emconfig.yaml.
model_config:
oc_svm:
kernel: 'rbf'
nu: 0.05
gamma: 'auto'
drift_detection:
det1_persistence: 10 # janelas consecutivas
det2_alpha: 0.001 # p-value threshold KS
adaptation:
a1_retraining_period: 50 # janelas
a2_buffer_size: 20 # instâncias recentes
ValidaçãodeIntegridadedeDados – Osdatasetsforam integrados comchecksumsSHA-256 para garantir
que versões idênticas sejam processadas em todas as repetições. Adicionalmente, umvalidation_gate.py
verificaa prioria qualidade dos dados (mínimo 1000 janelas/cenário, desequilíbrio de classes< 10 %) antes de
chamar apipelinede treino.
Documentação de Ambiente – Um ficheiroENVIRONMENT.mdestá disponível no repositório, detalhando:

Versão mínima de Python exigida (3.11+);
Todos os passos de instalação (clone,docker build,virtual environment setup);
Instruções de execução (ponto de entrada único:python run_full_pipeline.py);
Localização deoutputsesperados (results/metrics/,results/figures/).
3.4.5 Contextualização Industrial e Viabilidade Operacional
Embora o protótipo desenvolvido valide a prova de conceito técnica, a transição para um chão de fábrica real exige
uma reflexão crítica sobre as lacunas normativas, os requisitos de conformidade regulatória e os ajustes arquiteturais
necessários para garantir o alinhamento com práticas industriais consolidadas.
Conformidade com Normas Internacionais – A arquitetura proposta inscreve-se no domínio de
Manutenção Preditiva e Segurança Funcional, intersectando diversas normas internacionais de referência:

ISO 13374 (Condition Monitoring and Diagnostics): Esta norma estabelece o quadro de referência para
sistemas de deteção de falhas e monitorização contínua. O projeto cumpre os níveis 1 (aquisição de dados)
e 2 (processamento local), mas carece de implementação completa do nível 3 (fusão estratégica e decisão
tática).
IEC 61508 (Functional Safety of Electrical/Electronic/Programmable Systems): Para aplicações críticas
em chão de fábrica, é obrigatório certificar a integridade funcional da camada de deteção, especificando a
capacidade de diagnosticar falhas sistemáticas com probabilidades de falha certificadas (SIL -Safety Integrity
Level). O sistema atual não contempla mecanismos de auto-diagnóstico de falhas de sensores ou corrupção
desoftware.
ISO 55000 (Asset Management): Exigência de rastreabilidade de históricos de manutenção, ciclos de vida
de componentes e correlação com métricas financeiras de disponibilidade. Apipelineatual regista apenas
métricas de desempenho preditivo e não integra sistemas de gestão de ativos corporativos (ERP/CMMS).
Lacunas Normativas Identificadas e Ajustes Necessários:
Mecanismos de Validação de Sensores : Em ambiente industrial, toda a instrumentação está sujeita
a falhas silenciosas (driftde calibração, degradação do transdutor). A arquitetura proposta carece de um
módulo de auto-teste de sensores (BIST -Built-In Self-Test) que verifique, periodicamente, a plausibilidade
das leituras contra modelos físicos esperados.
Segurança Funcional e Certificação : Para aplicações onde a falha de deteção implique risco humano
ou dano financeiro severo, o sistema deve ser certificado sob IEC 61508 SIL 2 ou superior. Isto exige,
redundância de sensores, watchdogsdesoftware, recuperação automática após falhas, e auditoria
independente de código.
Integração com CMMS/ERP : Sistemas reais operam em conjunto com ferramentas deComputerized
Maintenance Management Systems(CMMS) como Maximo ou Infor EAM. Apipelineatual gera métricas
isoladas em CSV; seria necessário implementar APIsRESTfulou conectores demiddlewarepara ingestão em
tempo real destes sistemas.
Escalabilidade Horizontal : O protótipo foi validado numa única máquina (Smart Fan). Uma
implementação fabril típica monitoriza 50 a 200 equipamentos simultaneamente. Isto exigiria: (i)
arquitetura em nuvem (Edge+Cloud Sync), (ii) replicação de modelos para cada classe de equipamento,
(iii) orquestração distribuída tipo Kubernetes.
Conformidade de Privacidade (GDPR) : Os dados sensoriais contêm informações de processo que
podem ser considerados dados comercialmente sensíveis. Apipelinedeveria implementar mecanismos de
anonimização, encriptação em trânsito (TLS 1.3) e retenção configurável delogs.
Análise Crítica de Aplicabilidade Prática:
A estratégia A2 (Fine-tuning) provou ser energeticamente viável naEdge, com umSpeedupde 19.3x. No entanto,
o seu colapso preditivo (32.78%F1-Score) devido aoCatastrophic Forgettingtorna-a inviável para operação real sem
mitigação. Um sistema que falha sistematicamente 67% das deteções de anomalias não seria aceite por operadores
de chão de fábrica, independentemente da sua eficiência energética.
A transição para produção exigiria, portanto:

[label=(v)]
validação em múltiplos equipamentos reais com durações de teste de pelo menos 6 meses de operação
contínua;
2.comparação estatística com sistemas comerciais consolidados;
3.certificação por consultores de segurança funcional independentes.
Enquanto o projeto demonstra elevada relevância científica e inovação na deteção dedriftemEdge Computing,
a sua viabilidade operacional imediata em chão de fábrica permanece condicionada à resolução do problema de
Catastrophic Forgettinge à conformidade com normas de segurança funcional internacionais. É recomendado,
portanto, uma fase de piloto num ambiente industrial controlado, antes de qualquer implantação em produção em
larga escala.
4 Avaliação Experimental e Discussão de Resultados
Neste capítulo, apresentam-se e discutem-se os resultados da avaliação experimental do sistema proposto, com
especial enfoque na capacidade de deteção de anomalias e na eficiência da adaptação autónoma do modelo na
periferia da rede (Edge). A discussão está estruturada de forma progressiva: inicia-se com a validação da pipeline
de dados, evolui para a avaliação do impacto físico dodrift, e culmina na análise quantitativa de desempenho e do
custo computacional das estratégias de adaptação suportadas por testes de significância estatística.

4.1 Análise de Variabilidade Estatística e Dispersão
A injeção de ruído exógeno e a alteração deliberada das condições operacionais do sistema induzem modificações
mensuráveis no comportamento estatístico dos dados. Como ilustrado na Figura 10 , o cenárioIn-Control(D0)
caracteriza-se por uma dispersão extremamente compacta. Em contrapartida, os cenários com anomalias injetadas
(D1 a D4) evidenciam um aumento considerável na variância total do sinal, acompanhado pelo surgimento de
múltiplosoutliers. Estealargamentoprogressivovalidaexperimentalmentequeamonitorizaçãocontínuademétricas
de dispersão estatística de ordem superior (AccXStd) constitui um indicador preliminar robusto para a sinalização
precoce dedrift.

Figure 10: Distribuição da variância do sinal (AccXStd) por cenário, evidenciando o aumento da dispersão perante
a ocorrência de drift.

4.2 Comportamento Temporal e Características Latentes.
A transição entre regimes manifesta-se por quebras abruptas no perfil energético. A Figura 11 ilustra a atenuação
severa na amplitude RMS do sinal de aceleração (AccXRMS) aquando da transição para o cenário D2.

Simultaneamente, a frequência de pico fundamental oscila consoante as condições operacionais (Figura 12 ),
traduzindo com fidelidade as alterações físicas na máquina.

Figure 11: Série temporal evidenciando a quebra
abrupta na energia do sinal (AccXRMS) (D0 para D2).

Figure 12: Variação da frequência de pico fundamental
perante diferentes condições operacionais.
Para compreender o impacto multidimensional destas transformações, aplicou-se a Análise de Componentes
Principais (PCA). A Figura 13 demonstra que o cenário D0 se organiza numclusterdenso, enquanto os cenários D3
e D4 se projetam em regiões espaciais completamente exteriores à fronteira de decisão validada no treino inicial.

Figure 13: Análise PCA ilustrando a separação e o desvio dos cenários de drift no espaço de características latentes.

4.3 Desempenho dos Mecanismos de Deteção
A Tabela4.1sumaria otrade-offentre o DET1 e o DET2. Embora o DET2 registe um atraso médio global inferior
(23.00 janelas), a sua maior prontidão é alcançada à custa de uma acentuada vulnerabilidade ao ruído (30 falsos
alarmes). O DET1 provou ser imune a flutuações normais, exibindo máxima estabilidade.
A Figura 14 detalha o atraso de deteção segregado por anomalia. Exceptuando o cenário D3, o DET1 exibe um
tempo de resposta altamente competitivo.

Table 4.1 Comparação de Desempenho dos Algoritmos de Deteção deDrift.
Detetor Atraso Médio (Janelas) Falsos Alarmes (D0) Estabilidade
DET1 26.79 0 Máxima
DET2 23.00 30 Baixa
Figure 14: Atraso de deteção discriminado por cenário de anomalia.
4.3.1 Validação Estatística do Atraso de Deteção
Para determinar se a antecipação temporal do mecanismo é estatisticamente significativa, aplicou-se o teste não-
paramétrico deWilcoxon signed-rank. Os resultados consolidados na Tabela4.2atestam empiricamente que o
mecanismo DET1 obteve uma velocidade de isolamento superior de forma transversal (p-value< 0. 001 ), provando
ser consistentemente mais célere em anomalias térmicas e desvios compostos do que o DET2.

Table 4.2 Análise de significância estatística (Teste de Wilcoxon) para o atraso de deteção.
Cenário Diferença Média (DET1 - DET2) [Janelas] p-value Significância
D1 (Desvio Térmico) -10.0 0.000108 ***
D2 (Desvio Mecânico) -3.0 0.000108 ***
D4_D1eD2 -10.0 0.000108 ***
D4_D2eD3 -6.0 0.000108 ***
4.4 Eficácia das Lógicas de Adaptação e Tempo de Recuperação.
Uma vez diagnosticado o desvio, os mecanismos de mitigação são acionados. Contudo, a robustez do classificador
estático base (A0) degrada-se substancialmente perante a severidade dos cenários, conforme exposto na Tabela
4.3.

Table 4.3 Degradação do desempenho preditivo (F1-Score) do modelo estático base (A0).
Cenário de Drift Precision Recall F1-Score (A0) Degradação Relativa
D0 (Operação Normal) 0.98 0.96 0.97 Baseline
D1 (Desvio Térmico) 0.92 0.85 0.88 -9.2%
D2 (Desvio Mecânico) 0.89 0.81 0.85 -12.3%
D3 (Deg Sensor) 0.61 0.48 0.54 -44.3%
D4 (Drift Combinado) 0.45 0.32 0.37 -61.8%
Paradoxalmente, a Tabela4.4evidencia que o modelo sem adaptação (A0) manteve a melhor performance
global (85.76%), ao passo que a aplicação autónoma das estratégias de retreino total (A1) efine-tuning(A2) resultou
num declínio acentuado.

Table 4.4 Avaliação Global do F1-Score Preditivo entre estratégias.
Estratégia de Aprendizagem F1-Score Global (Weighted)
A0 (Sem Adaptação) 85.76%
A1 (Retreino Total) 58.10%
A2 (Fine-tuning) 32.78%
Estes resultados atestam o fenómeno clássico deCatastrophic Forgetting. Sem umaValidation Gateque retenha
a memória saudável do sistema, o modelo efetuaoverfittingà própria anomalia. A raiz desta degradação é dissecada
nas Matrizes de Confusão (Figura 15 ). Ao aplicar a adaptação, o classificador absorve a anomalia, o que se traduz
numa explosão de falsos negativos aquando do retorno à operação normal (observável na quebra prolongada patente
na Figura 16 e no tempo de recuperação inercial ilustrado na Figura 17 ).
Surge aqui uma questão fulcral na discussão desta arquitetura adaptativa: de que serve dispor de uma
estratégia define-tuningcapaz de processar atualizações com um ganho de velocidade (Speedup) de 19.3x
(conforme detalhado na Secção4.5) se esta falha o seu propósito analítico basilar ao colapsar a eficácia preditiva
para uns dramáticos 32.78% deF1-Score? Esta gritante discrepância comprova que, embora a celeridade
computacional de A2 valide com sucesso a prova de conceito técnica no ecossistemaEdge, a abordagem carece
inteiramente de viabilidade operacional imediata. Sem a integração de uma lógica de retenção de conhecimento

ancestral que sirva de âncora geométrica — como a heurística de Replay Buffersugerida nas linhas de
investigação futura —, o modelo torna-se instável e inapto para aplicação real num chão de fábrica.

(a) Modelo Base A0 (b) Modelo comFine-tuningA2
Figure 15: Matrizes de confusão ilustrando a ocorrência deCatastrophic Forgetting.
Figure 16: Evolução temporal do F1-Score evidenciando
o colapso nas lógicas adaptativas. Figure 17: Tempo de recuperação (Recovery Time) em

janelas amostrais.
4.5 Impacto Energético e Latência Computacional na Periferia
Apesar dos desafios relacionados com o esquecimento algorítmico, a viabilidade naEdgeestá subordinada a
constrangimentos físicos. A Figura 18 illustrates que a estratégia A1 é energeticamente proibitiva (503.19 J),
enquanto ofine-tuningA2 se mantém idêntico àbaseline.
A sobrecarga computacional que dita esta penalização térmica está consolidada na Tabela4.5. A estratégia A1
impõe um estrangulamento latente médio superior a 261 ms. Inversamente, a estratégia A2 fornece umSpeedup
brutal de 19.3x, resolvendo a adaptação em apenas 13.56 ms. A dispersão desta latência é claramente visível no
Boxplot(Figura19a).
Para ilustrar este compromisso visualmente, a Figura19bapresenta a Frente de Pareto. O gráfico evidencia
que a estratégia A2 domina o espaço de soluções viáveis num ecossistema de IoT, minimizando o effort de latência.

Figure 18: Consumo energético cumulativo comparando A0, A1 e A2.
Table 4.5 Avaliação do custo computacional e ganho de desempenho (Speedup) das estratégias naEdge.
Estratégia Latência Média (ms) Máximo (ms) Desvio Padrão Speedup (vs. A1)
Sem Adaptação (A0) 0.00 0.00 0.00 –
Retreino Total (A1) 261.43 324.58 7.77 1.0x
Fine-tuning (A2) 13.56 18.45 5.58 19.3x
4.5.1 Validação Estatística de Significância Computacional
Para assegurar que o contraste computacional traduz variações rigorosas dehardwaree não flutuações estocásticas
do microprocessador, aplicou-se o testeWilcoxon Signed-Rankà latência. A análise inferencial gerou ump-value<

001 , ratificando de forma irrefutável a superioridade do escalonamento A2 comparativamente à rigidez introduzida
pelo recálculo total (A1). Este rigor valida definitivamente a via dofine-tuningleve como a arquitetura ideal para a
Indústria 4.0.
(a) Distribuição da Latência Computacional (b) Frente de Pareto

Figure 19: Análise comparativa do custo temporal e de convergência entre as estratégias.
5 Conclusão e Trabalho Futuro
O presente trabalho consolidou uma abordagem técnica e científica para a deteção deconcept drifte adaptação
de modelos de manutenção preditiva em contextos deEdge Computing. Através da definição rigorosa de uma
arquitetura distribuída em três camadas e da execução de protocolos experimentais controlados de injeção de
falhas, foi possível avaliar o comportamento de sistemas de monitorização em ambientes industriais simulados.

As decisões adotadas refletem ostrade-offscentrais inerentes à computação na periferia da rede, nomeadamente
o equilíbrio entre a precisão preditiva, a latência operacional e a eficiência energética. A avaliação quantitativa destas
métricas produziu conclusões fundamentais sobre a viabilidade e os desafios da implementação de algoritmos de
aprendizagem contínua não supervisionada em microcontroladores e dispositivos com recursos limitados.

5.1 Síntese das Contribuições e Conclusões Principais.
A principal contribuição prática deste estudo reside na demonstração empírica da superioridade dofine-tuning
(Estratégia A2) como o mecanismo de adaptação mais adequado para arquiteturasEdge. Em nítido contraste
com o retreino periódico global (Estratégia A1), que se revelou uma operação energeticamente asfixiante e com
elevadooverheadcomputacional, a estratégia A2 reduziu drasticamente o custo temporal e térmico da intervenção.
Estes resultados atestam de forma categórica que as abordagens leves de recalibração localizada de tensores são
a via computacional e energeticamente mais sustentável para implementações escaláveis de inteligência artificial
distribuída na Indústria 4.0.

5.2 Limitações Identificadas e Ameaças à Validade
Apesar da notável eficiência operacional e do baixo custo latente da estratégia define-tuning, a avaliação contínua
expôs uma vulnerabilidade crítica intrínseca à aprendizagem não supervisionada: o colapso do desempenho
preditivo devido aoCatastrophic Forgetting(esquecimento catastrófico). OF1-Scoreglobal das estratégias de
adaptação degradou-se substancialmente perante anomalias sustentadas. Constatou-se que o modelo, devido à
ausência de umaValidation Gateque funcionasse como um filtro prévio, atualizou as suas fronteiras de decisão
baseando-se cegamente nas janelas instáveis recém-adquiridas. Ao absorver a anomalia injetada como se
representasse o novo regime normal de operação, o algoritmo corrompeu o espaço latente da assinatura vibratória
saudável, perdendo a capacidade de discernir um retorno ao estado operacionalIn-Control.

Para além da degradação porCatastrophic Forgetting, importa referir uma ameaça à validade externa do modelo
intimamente ligada à montagem experimental. Embora o ambiente laboratorial adotado possibilite um isolamento
rigoroso e reprodutível das variáveis dedriftinduzido, a simulação suportada numa planta física isolada (protótipo

Smart Fan) não traduz na íntegra a complexidade vibratória de um chão de fábrica real. Num ambiente industrial
genuíno, os equipamentos estão sujeitos astressesexternos partilhados, tais como vibrações parasitas provenientes
de máquinas adjacentes e fenómenos de ressonância estrutural não previstos. Assumir estas restrições físicas de
hardwarenão invalida a prova de conceito, mas evidencia que o limiar de ruído operacional natural será superior,
exigindo calibrações futuras do modeloOne-Classperante estas interferências multidimensionais.

5.3 Linhas de Investigação Futura.
Para colmatar as limitações identificadas e possibilitar a utilização plena do baixooverheaddofine-tuningsem
a penalização associada ao esquecimento catastrófico, o trabalho futuro deverá focar-se no desenho de lógicas
heurísticas leves para a conservação de memória latente na periferia.
Neste sentido, sugere-se a incorporação arquitetural de umReplay Buffer(memória de repetição episódica)
que preserve ativamente um subconjunto validado de dados antigos referentes ao estado operacional saudável. Ao
forçar o modelo a revisitar e reavaliar estas amostras de referência em simultâneo com os novos dados dedrift
durante o ciclo de adaptação incremental, será possível ancorar matematicamente a fronteira de decisão original.
Esta abordagem de validação cruzada contínua promete atenuar severamente oCatastrophic Forgetting, garantindo
uma resiliência preditiva de longo prazo sem onerar o consumo energético de dispositivosEdge.

Bibliografia
[1]Eyad Megdadi, Azza Mohamed, and Khaled Shaalan. Machine learning-driven best–worst method for predictive
maintenance in industry 4.0. Automation, 6(4):91, December 2025. ISSN 2673-4052. doi: 10.3390/
automation6040091.

[2]Shreya Prabhudesai, Shruti Patil, Satish Kumar V C, Pooja Kamat, and Ketan Kotecha. An explainable predictive
maintenance strategy for multi-fault diagnosis of rotating machines using multi-sensor data fusion. Decision
Analytics Journal, 10:100425, February 2024. doi: 10.1016/j.dajour.2024.100425.

[3]Jibinraj Antony, Dorotea Jalušić, Simon Bergweiler, Ákos Hajnal, Veronika Žlabravec, Márk Emődi, Dejan
Strbad, Tatjana Legler, and Attila Csaba Marosi. Adapting to changes: A novel framework for continual
machine learning in industrial applications. Journal of Grid Computing, 22(4), 2024. ISSN 1570-7873. doi:
10.1007/s10723-024-09785-z.

[4]Łukasz Pawlik. Mitigating concept drift in wind turbine prognostics using dynamic feature engineering and
chronological validation. IEEE Access, 14:44491–44502, 2026. ISSN 2169-3536. doi: 10.1109/ACCESS.
2026.3676340.

[5]Rajesh Shah, Vikram Mittal, and Michael Lotwin. Recent advances in vibration analysis for predictive
maintenance of modern automotive powertrains.Vibration, 8(4):68, December 2025. ISSN 2571-631X. doi:
10.3390/vibration8040068.

[6]A. Esteban, A. Cano, S. Ventura, and A. Zafra. Simultaneous fault prediction in evolving industrial environments
with ensembles of hoeffding adaptive trees. 55, 2025. ISSN 0924-669X. doi: 10.1007/s10489-025-06786-7.

[7]Evolving strategies in machine learning: A systematic review of concept drift detection. 15:1–24, December

doi: 10.3390/info15120786.
[8]Govind Vashishtha, Sumika Chauhan, and Merve Ertarğın. A metric-driven evaluation framework for remaining
useful life prognosis with quantified uncertainty. Sensors, 26(7):2230, January 2026. ISSN 1424-8220. doi:
10.3390/s26072230.

[9]Farzam Farbiz, Saurabh Aggarwal, Tomasz Karol Maszczyk, Mohamed Salahuddin Habibullah, and Brahim
Hamadicharef. Reliability-improved machine learning model using knowledge-embedded learning approach for
smart manufacturing.Journal of Intelligent Manufacturing, 36(7):4941–4962, October 2025. ISSN 1572-8145.
doi: 10.1007/s10845-024-02482-4.

14
P0 7 : DriftSense-PM: Concept Drift Detection and Adaptive Predictive Maintenance at
the Edge
Proponent: Flávio de Oliveira Silva – flavio@di.uminho.pt
Areas: Internet of Things; Edge Computing; Machine Learning for Networks; Smart Industry
Context and Objectives
Predictive maintenance (PM) is a key enabler of Industry 4.0, allowing early detection of
failures and reduction of downtime through continuous monitoring of industrial equipment
using IoT sensors. However, most PM solutions implicitly assume stationary data distributions,
which rarely holds in real industrial environments. Changes in operating regimes, sensor
aging, environmental conditions, or mechanical wear introduce concept drift, significantly
degrading model accuracy over time.
This project fits naturally within the Internet / Next Generation Networks Engineering Project
course by combining IoT sensing, edge computing, machine learning, and system evaluation
under realistic constraints. It applies knowledge domains such as IoT architecture, data
analytics, edge/cloud cooperation, and performance evaluation.
The project’s main objective is to design, implement, and experimentally evaluate a drift-
aware predictive maintenance pipeline running on real edge devices, demonstrating how drift
detection and adaptation strategies improve reliability while respecting latency and energy
constraints.
Obj1 – Design a predictive maintenance pipeline using real industrial sensors.
Obj1 - esign a predictive maintenance pipeline based on real industrial sensors deployed at
the edge.
Obj2 - Implement and compare multiple concept drift detection mechanisms under controlled
drift scenarios.
Obj3 - Implement and evaluate model adaptation strategies suitable for edge and near-edge
execution.
Obj4 - Quantify the impact of drift awareness on prediction accuracy, detection delay, latency,
and energy consumption.
Proposed Experiment Set
EXP- 1 – Baseline predictive maintenance without drift awareness.
EXP- 2 – Predictive maintenance with drift detection but no adaptation.
EXP- 3 – Drift-aware predictive maintenance with periodic and incremental adaptation.
Anexo I - Proposta de Projeto
15
Experiment Matrix (Summary)
Experiments vary drift type (temperature, vibration, sensor bias), detector (none, statistical,
distribution-based), and adaptation strategy (none, retraining, incremental). Metrics include
F1-score, drift detection delay, false alarms, inference latency, and energy per decision.
Expected Results
The expected outcomes of the project include:
A working predictive maintenance prototype using real sensor data collected from
industrial-grade hardware.
An experimental dataset with annotated drift scenarios (e.g., temperature, vibration,
sensor bias).
A comparative evaluation of static vs drift-aware PM models.
Quantitative results showing trade-offs between accuracy, detection delay,
adaptation cost, and energy consumption.
Observations / Notes (practical details to include in the paper)
Controlled drift injection is essential for repeatability. Energy measurements should
be explicitly reported. Edge/cloud cooperation must be clearly documented.
Hardware: Arduino Pro Smart Industry Predictive Maintenance Kit and Raspberry Pi
Emphasis should be placed on repeatability and controlled drift injection.
Energy measurements should be explicitly reported.
Planning
Task 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Presentation
and selection
of topics
T1: Planning
and scope
definition
T2: State-of-
the-art study
T3:
Requirements
gathering /
Solution
architecture
T4: Data
acquisition +
baseline PM
pipeline
Anexo I - Proposta de Projeto
16
T5: Drift
injection +
drift detection
T6:
Adaptation
strategies +
evaluation
campaign
T7: Results
analysis +
ablations +
replication
package
T8: Paper
writing + final
polishing
Final
Evaluation
Related Work and References
[1] J. Gama et al., “A survey on concept drift adaptation,” ACM Computing Surveys, 2014,
doi: 10.1145/2523813.
[2] Y. Maher et al., “Survey on Deep Learning applied to predictive maintenance,”
International Journal of Electrical and Computer Engineering (IJECE), 2020, doi:
10.11591/ijece.v10i6.pp5592-559.
Anexo I - Proposta de Projeto
A Apêndice A
A.1 Injeção de Ruído Sintético (Cenário D3)
O trecho de código abaixo documenta a lógica de geração sintética para a emulação de falhas nos transdutores
(D3), garantindo a exata reprodutibilidade do ruído Gaussiano e da introdução dobiasabordados na Secção 4.

Definicoes do Ruido e do Bias
fator_de_ruido = 0.15
bias_offset = 20.0 # Simula a perda de calibracao com o tempo
np.random.seed(42) # Mantido para garantias de reprodutibilidade

colunas_vibracao = ['AccX', 'AccY', 'AccZ']

for eixo in colunas_vibracao:

Calcula o ruido gaussiano (15% do desvio padrao original)
sigma = df_d3[eixo].std() * fator_de_ruido
ruido_gaussiano = np.random.normal(0, sigma, size=len(df_d3))

# Soma o ruido e o bias, arredondando o output
df_d3[eixo] = (df_d3[eixo] + ruido_gaussiano + bias_offset).round(1)