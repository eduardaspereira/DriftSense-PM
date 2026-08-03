Coisas que temos de adicionar/corrigir no artigo:

- Colocar o artigo com 6 paginas 

- Em vez de usarmos o termo fine-tuning no One-Class SVM deveriamos dizer Unsupervised Localized Re-fitting

- Falta abordarmos na Secção III.B : Frequência de amostragem, Tamanho da Janela, Sobreposição (Overlap), FFT, Normalização (escalonamento StandarScaler)

- Dois erros reais cientificamente provados e que nao solucionamos e deveriamos apresentar ao professor:

    - Inabilidade do DET2

        - Explicação do Problema

            - O DET2 gerou 30 falsos alarmes em regime saudável, enquanto o DET1 manteve estabilidade máxima

            - Foi comprovado que detetores puramente estatísticos baseados em dados brutos sofrem de hipersensibilidade

        - Solução

            - IMplementar uma lógica de suavização por janela deslizante e duplo limiar de histerese sobre os p-values do teste KS

    - Catastrophic forgetting

        - Explicação do Problema

            - O A2 alcancou uma eficiência de hardware muito positiva, porém um F1-Score global muito baixo (32.78%)

            - Ao reajustarmos o One Class SVM a usar um buffer de 20 amostras de drift recente, o modelo define a avaria como o novo estado normal, fazendo com que quando a máquina regresse ao estado nominal (D0), este seja irremediavelmente apagado.

        - Solução

            - Streaming Continual Industrial Learning (possivelmente enviavel na Edge) / Nominal Anchor Rehearsal (NAR - Repetição de Memória Episódica)

- o prof vai verificar o artigo e temos de esperar para fazer uma v2 do draft mais forte dq a primeira

- Melhorias:

    - 1. Inconsistência Crítica nas Figuras

        - O texto na Secção IV-D faz referência à Fig. 4 indicando que esta contém "confusion matrices illustrating the occurrence of Catastrophic Forgetting". No entanto, os gráficos efetivamente apresentados sob a Fig. 4 são um gráfico de barras referente à análise de atraso de deteção e um boxplot do perfil de latência computacional

        - Ação: Devem substituir as imagens pelas matrizes de confusão corretas para ilustrar a diferença entre os 680 Falsos Negativos e 50 Falsos Positivos do modelo A0 face ao colapso do modelo A2. Os gráficos de latência e deteção que lá estão inseridos parecem pertencer a secções anteriores (como a Secção IV-B ou IV-C).

    - 2. Completude da Solução Proposta

        - A Falta do Replay Buffer: O artigo conclui que a estratégia de adaptação local não supervisionada (A2) é operacionalmente inviável de forma isolada, dado que a sua aplicação faz o F1-Score cair drasticamente para 32.78%. Como solução, sugerem a incorporação de um Episodic Replay Buffer no trabalho futuro.

        - Ação: Para uma publicação de alto impacto, identificar um paradoxo muitas vezes não é suficiente. Se a tua colega conseguiu realizar testes extensivos na arquitetura, seria de enorme valor acrescentar a implementação desse Replay Buffer já neste artigo. Provar que a adição de memória episódica estabiliza o modelo A2 elevaria a vossa submissão de um "estudo empírico de um problema" para a "proposta de uma solução completa".
    
    - 3. Rigor Metodológico e Justificações

        - Limiares Empíricos: No mecanismo DET1, definem o período de persistência para as anomalias como $P=10$ janelas de amostragem, referindo que este valor foi determinado empiricamente.

        - Ação: Revistas rigorosas exigem justificação para estes limiares. Adicionem uma breve frase ou referência estatística explicando como chegaram a esse número (e.g., uma análise de sensibilidade prévia ou características físicas dos transitórios mecânicos do motor).

        - Formalização Matemática: A definição formal do concept drift apresentada na introdução, $P_{t_{0}}(X,y)\ne P_{t_{1}}(X,y)$, está correta. Contudo, garantir a definição explícita das variáveis (onde $X$ representa o espaço de features e $y$ a variável alvo) imediatamente a seguir à equação aumentará a clareza formal do texto. 

    - 4. Análise de Consumo Energético

        - Comparação do Baseline: A Figura 3 e a respetiva análise referem que o consumo de energia cumulativo do modelo A0 (sem adaptação) é de 223.32 J, e o do modelo A2 é de 223.73 J.

        - Ação: Seria interessante adicionar uma breve discussão sobre se o ligeiro aumento de 0.41 J tem alguma significância estatística ao longo de múltiplos ciclos de teste ou se é apenas ruído térmico/elétrico residual.

- 