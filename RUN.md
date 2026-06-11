Descrição
---------
Conjunto de comandos exactos para executar o fluxo principal do DriftSense-PM.
Este ficheiro serve como referência rápida para reproduzir a solução depois de
o ambiente já estar configurado.

Autores
-------
- Eduarda Pereira
- Gonçalo Ferreira
- Gonçalo Magalhães

Pré-condições
-------------
- Ambiente Python já activo.
- Dependências instaladas.
- `configs/driftsense_dataset/config.yaml` ajustado aos caminhos locais.
- Dados raw disponíveis nas localizações configuradas.

Comandos exactos
---------------
Executar a partir da pasta `scripts/driftsense_dataset/`:

```bash
cd scripts
python3 feature_engineering.py
python3 train_baseline_full.py
python3 master_script.py
python3 statistical_analysis.py
python3 generate_thesis_plots.py
```

Ordem recomendada
-----------------
1. `feature_engineering.py`
2. `train_baseline_full.py`
3. `master_script.py`
4. `statistical_analysis.py`
5. `generate_thesis_plots.py`

Saídas principais
-----------------
- Modelos em `models/driftsense_dataset/`.
- Métricas em `results/driftsense_dataset/metrics/`.
- Figuras em `results/driftsense_dataset/figures/`.
- Ficheiros processados em `data/driftsense_dataset/processed/`.
