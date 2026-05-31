Descrição
---------
Guia técnico de reprodutibilidade do projecto DriftSense-PM. Este ficheiro
resume os pré-requisitos de ambiente, a instalação das dependências, a
configuração central e os passos necessários para reproduzir os resultados.

Autores
-------
- Eduarda Pereira
- Gonçalo Ferreira
- Gonçalo Magalhães

Ambiente de execução
--------------------
- Linux (Ubuntu 20.04+ recomendado) ou Raspberry Pi OS.
- Python 3.10+.
- Ambiente isolado com `conda` ou `virtualenv`.

Instalação
----------
Conda:

```bash
conda env create -f env/environment.yml
conda activate driftsense-pm
```

pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

Configuração
------------
Edite `configs/config.yaml` antes de correr a pipeline. Os campos mais
importantes são:

- `paths.raw_data_dir`
- `paths.processed_dir`
- `paths.models_dir`
- `paths.results_dir`
- `system.sampling_rate_hz`
- `feature_engineering.window_size`
- `feature_engineering.step_size`

Reprodução completa
-------------------
1. Activar o ambiente preparado.
2. Executar a extracção de features:

```bash
cd scripts
python3 feature_engineering.py
```

3. Treinar o baseline:

```bash
python3 train_baseline_full.py
```

4. Correr a avaliação factorial:

```bash
python3 master_script.py
```

5. Gerar a análise estatística:

```bash
python3 statistical_analysis.py
```

6. Gerar as figuras finais:

```bash
python3 generate_thesis_plots.py
```

Saídas esperadas
----------------
- `models/baseline_model.pkl`
- `models/scaler.pkl`
- `results/metrics/`
- `results/figures/`
- `data/processed/`
