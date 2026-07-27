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



