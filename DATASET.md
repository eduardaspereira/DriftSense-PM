# DriftSense-PM: Dataset v1.0

## 1. Visão Geral
Este dataset foi recolhido para avaliar pipelines de manutenção preditiva cientes de *concept drift* na *Edge*. Os dados foram gerados usando um Arduino Pro Smart Industry Predictive Maintenance Kit e o sensor Nicla Sense ME, transmitidos para um Raspberry Pi 5.

## 2. Protocolo de Injeção de Falhas (Drift Taxonomy)
A recolha seguiu a taxonomia do projeto, com um total de 1200 janelas de baseline. Os ficheiros foram mapeados de acordo com a nomenclatura oficial.

* **D0 (No Drift - Baseline):** O sensor Nicla Sense ME foi acoplado com abraçadeiras a um motor DC com ventoinha, repousando sobre uma estrutura fixa. Operação estabilizada a 50% da potência máxima durante 1200 janelas.
* **D1 (Covariate Drift - Temperature):** No mesmo setup físico de D0, foi aplicado um secador de cabelo direcionado ao equipamento para induzir o aumento gradual da temperatura ambiental e do hardware.
* **D2 (Mechanical Drift - Mounting):** *Desvio do Plano Inicial:* Omitido neste dataset v1.0. A decisão de usar abraçadeiras fixas impossibilitou uma alteração controlada e reprodutível da montagem sem comprometer o isolamento dos restantes testes.
* **D3 (Operational Drift - Regime):** Modificação do ciclo de operação. A potência do motor DC (RPM) foi aumentada de 50% (baseline) para 75%, alterando o padrão de vibração estrutural. (Nota interna: gravado originalmente como D2).
* **D4 (Sensor Degradation - Noise):** Injeção matemática controlada de ruído Gaussiano sobre o sinal estabilizado, emulando o envelhecimento e a perda de calibração dos sensores (bias shift). (Nota interna: gravado originalmente como D3).
* **D5 (Combined Drift - Realistic):** Fusão de cenários. Inclui um sub-cenário combinando D1 (Temperatura) com D3 (Regime), e um sub-cenário combinando os efeitos operacionais com D4 (Ruído Gaussiano).