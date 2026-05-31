# DriftSense-PM

Descrição
---------
Manual de reprodutibilidade técnica do projecto DriftSense-PM. Este repositório
agrega o código, os dados processados e os artefactos necessários para reproduzir
o pipeline completo de extração de características, treino de modelos, avaliação
factorial, análise estatística e geração de gráficos.

Autores
-------
- Eduarda Pereira
- Gonçalo Ferreira
- Gonçalo Magalhães

Pré-requisitos
--------------
- Sistema operativo Linux (Ubuntu 20.04+ recomendado) ou Raspberry Pi OS.
- Python 3.10 ou superior.
- `git` para obter o código-fonte.
- `conda` ou `virtualenv` para isolamento do ambiente.
- `docker` apenas se pretender usar o fluxo de contentorização.

Instalação de dependências
--------------------------
O projecto disponibiliza dependências em `env/environment.yml` e em
`env/requirements.txt`.

Conda:

```bash
conda env create -f env/environment.yml
conda activate driftsense-pm
```

pip + virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

Configuração técnica
--------------------
O ficheiro central é `configs/config.yaml`. Confirme, pelo menos, os seguintes
parâmetros antes de executar a solução:

- `paths.raw_data_dir`: origem dos CSV raw.
- `paths.processed_dir`: destino dos ficheiros de features.
- `paths.models_dir`: destino dos modelos treinados.
- `paths.results_dir`: destino de métricas, relatórios e figuras.
- `system.sampling_rate_hz`: frequência de amostragem usada no processamento.
- `feature_engineering.window_size` e `feature_engineering.step_size`.

Execução da solução
-------------------
1. Ativar o ambiente criado na secção anterior.
2. Mudar para a pasta dos scripts:

```bash
cd scripts
```

3. Executar a extracção de features:

```bash
python3 feature_engineering.py
```

4. Treinar o modelo baseline:

```bash
python3 train_baseline_full.py
```

5. Executar a avaliação factorial completa:

```bash
python3 master_script.py
```

6. Gerar análise estatística e tabelas finais:

```bash
python3 statistical_analysis.py
```

7. Gerar as figuras finais:

```bash
python3 generate_thesis_plots.py
```

Resultados esperados
--------------------
- Modelos: `models/baseline_model.pkl` e `models/scaler.pkl`.
- Métricas: `results/metrics/`.
- Figuras: `results/figures/`.
- Dados processados: `data/processed/`.

Documentação relacionada
------------------------
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) contém a reprodução técnica.
- [RUN.md](RUN.md) contém os comandos exactos de execução.
