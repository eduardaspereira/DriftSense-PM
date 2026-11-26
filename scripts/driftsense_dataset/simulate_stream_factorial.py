#!/usr/bin/env python3
"""
Descrição: Simulação em streaming com desenho fatorial 2x2 das estratégias de
           adaptação, separando o MECANISMO DE ATIVAÇÃO do CONJUNTO DE TREINO.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Responde à revisão P1 "Comparabilidade das estratégias A1 e A2" e
"Validade da comparação de latências".

                       | Treino HIST+BUF (D0 + 20)  | Treino BUF (20 recentes)
  ---------------------+----------------------------+-------------------------
  Periódico (cada 50)  | A1       (original)        | A1-buf   (novo)
  Por deteção (1x)     | A2-hist  (novo)            | A2       (original)

  + A0 (sem adaptação) como referência.

A lógica de streaming (prever -> detetar -> adaptar -> recuperar) replica
master_script.py. Diferenças deliberadas, todas documentadas:
  * D0 é pré-carregado em memória; o I/O de disco é medido à parte em
    benchmark_adaptation_size.py, para que a latência aqui seja só treino.
  * Cada atualização é registada individualmente (updates.csv), em vez de só
    a soma por cenário -> nº de atualizações, latência por atualização e
    custo acumulado ficam separados.
  * Todas as previsões janela-a-janela são gravadas (predictions.csv) para
    que qualquer métrica de qualidade seja recalculável.

Não requer hardware nem novos cenários. Correr no Raspberry Pi 5:
    python simulate_stream_factorial.py --reps 5 --cpu 3
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
META_COLS = ['Scenario', 'Timestamp', 'SysState', 'SampleCount']

# Valores por omissão = configs/driftsense_config.yaml (o mesmo que master_script lê).
cfg = {}
cfg_path = os.path.join(PROJECT_ROOT, "configs", "driftsense_config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
det_cfg = cfg.get('detectors', {})
ad_cfg = cfg.get('adaptation', {})

parser = argparse.ArgumentParser(description="Streaming 2x2: ativação x conjunto de treino")
parser.add_argument("--processed-dir", default=os.path.join(PROJECT_ROOT, "data/driftsense_dataset/processed"))
parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models/driftsense_dataset"))
parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results/driftsense_dataset/metrics/stream_factorial"))
parser.add_argument("--reps", type=int, default=5, help="Repetições (só a latência varia)")
parser.add_argument("--persistence", type=int,
                    default=det_cfg.get('det1_error_monitoring', {}).get('persistence', 10))
parser.add_argument("--alpha-ks", type=float,
                    default=det_cfg.get('det2_distribution_test', {}).get('alpha_ks', 0.001))
parser.add_argument("--ks-window", type=int,
                    default=cfg.get('feature_engineering', {}).get('window_size', 20))
parser.add_argument("--a1-interval", type=int,
                    default=ad_cfg.get('a1_periodic_retrain', {}).get('retrain_interval', 50))
parser.add_argument("--buffer-size", type=int,
                    default=ad_cfg.get('a2_lightweight', {}).get('buffer_size', 20))
parser.add_argument("--stability", type=int, default=5, help="Janelas normais seguidas = recuperado")
parser.add_argument("--nu", type=float, default=0.05)
parser.add_argument("--gamma", default="auto")
parser.add_argument("--cpu", type=int, default=None)
args = parser.parse_args()

if args.cpu is not None and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {args.cpu})

# Estratégia -> (gatilho, dados de treino)
STRATEGIES = {
    "A0":      (None,       None),
    "A1":      ("periodic", "HIST+BUF"),
    "A1-buf":  ("periodic", "BUF"),
    "A2":      ("detect",   "BUF"),
    "A2-hist": ("detect",   "HIST+BUF"),
}
DETECTORS = ["DET1", "DET2"]

d0_file = [f for f in os.listdir(args.processed_dir) if f.startswith('D0_')][0]
df_d0 = pd.read_csv(os.path.join(args.processed_dir, d0_file))
X_hist = df_d0.drop(META_COLS, axis=1, errors='ignore')
ref_vib = df_d0['AccX_RMS'].values
scenario_files = [d0_file] + sorted(
    f for f in os.listdir(args.processed_dir)
    if f.endswith('.csv') and f != d0_file and not f.startswith('D5'))

base_model = joblib.load(os.path.join(args.models_dir, 'baseline_model.pkl'))
base_scaler = joblib.load(os.path.join(args.models_dir, 'scaler.pkl'))


def refit(X):
    """Operação idêntica a adaptations.py (sem I/O)."""
    t0 = time.perf_counter()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = OneClassSVM(kernel='rbf', nu=args.nu, gamma=args.gamma)
    model.fit(Xs)
    return model, scaler, (time.perf_counter() - t0) * 1e3, int(model.support_vectors_.shape[0])


def simulate(fname, detector, strategy, rep):
    trigger, data = STRATEGIES[strategy]
    model, scaler = base_model, base_scaler
    df = pd.read_csv(os.path.join(args.processed_dir, fname))
    cols = [c for c in df.columns if c not in META_COLS]
    Xall = df[cols]

    detection_idx, consecutive = None, 0
    buffer_X, window_vib = [], []
    adapted_once = False
    is_recovering, rec_start, recovery_time, normal_run = False, None, None, 0
    version = 0
    updates, preds = [], []

    for i in range(len(df)):
        x_raw = Xall.iloc[[i]]
        buffer_X.append(x_raw.values[0])
        if len(buffer_X) > args.buffer_size:
            buffer_X.pop(0)
        window_vib.append(df.iloc[i]['AccX_RMS'])
        if len(window_vib) > args.ks_window:
            window_vib.pop(0)

        y_pred = int(model.predict(scaler.transform(x_raw))[0])
        preds.append((i, y_pred, version))

        # Deteção (igual a master_script: só até à 1.ª deteção)
        if detection_idx is None:
            if detector == 'DET1':
                consecutive = consecutive + 1 if y_pred == -1 else 0
                if consecutive >= args.persistence:
                    detection_idx = i
            elif len(window_vib) == args.ks_window:
                if ks_2samp(ref_vib, window_vib).pvalue < args.alpha_ks:
                    detection_idx = i

        # Adaptação
        fire = False
        if trigger == "periodic" and i > 0 and i % args.a1_interval == 0:
            fire = True
        elif trigger == "detect" and detection_idx is not None and not adapted_once:
            fire, adapted_once = True, True
        if fire:
            df_buf = pd.DataFrame(buffer_X, columns=cols)
            X_train = (pd.concat([X_hist[cols], df_buf], ignore_index=True)
                       if data == "HIST+BUF" else df_buf)
            model, scaler, lat, n_sv = refit(X_train)
            version += 1
            updates.append({"window_idx": i, "n_train": len(X_train),
                            "latency_ms": lat, "n_sv": n_sv, "model_version": version})
            if recovery_time is None:
                is_recovering, rec_start, normal_run = True, i, 0

        # Recuperação (igual a master_script)
        if is_recovering and recovery_time is None:
            normal_run = normal_run + 1 if y_pred == 1 else 0
            if normal_run >= args.stability:
                recovery_time = (i - args.stability + 1) - rec_start
                is_recovering = False

    return detection_idx, recovery_time, updates, preds, len(df)


upd_rows, run_rows, pred_rows = [], [], []
t0 = time.perf_counter()
for fname in scenario_files:
    scen = fname.split('_dataset')[0]
    for det in DETECTORS:
        for strat in STRATEGIES:
            for rep in range(1, args.reps + 1):
                d_idx, rec, ups, preds, n = simulate(fname, det, strat, rep)
                for u in ups:
                    upd_rows.append({"Scenario": scen, "Detector": det, "Strategy": strat,
                                     "Trigger": STRATEGIES[strat][0], "TrainData": STRATEGIES[strat][1],
                                     "Repetition": rep, **u})
                lats = [u["latency_ms"] for u in ups]
                run_rows.append({
                    "Scenario": scen, "Detector": det, "Strategy": strat, "Repetition": rep,
                    "n_windows": n, "detection_idx": d_idx, "n_updates": len(ups),
                    "n_train_per_update": ups[0]["n_train"] if ups else 0,
                    "latency_per_update_mean_ms": float(np.mean(lats)) if lats else np.nan,
                    "latency_cumulative_ms": float(np.sum(lats)),
                    "recovery_time": rec,
                })
                if rep == 1:  # previsões são determinísticas -> gravar uma vez
                    for i, y, v in preds:
                        pred_rows.append({"Scenario": scen, "Detector": det, "Strategy": strat,
                                          "window_idx": i, "y_pred": y, "model_version": v})
        print(f"  {scen} concluído")
elapsed = time.perf_counter() - t0

os.makedirs(args.out_dir, exist_ok=True)
df_upd, df_run, df_pred = pd.DataFrame(upd_rows), pd.DataFrame(run_rows), pd.DataFrame(pred_rows)
df_upd.to_csv(os.path.join(args.out_dir, "updates.csv"), index=False)
df_run.to_csv(os.path.join(args.out_dir, "runs.csv"), index=False)
df_pred.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

# ---- Latência: por atualização vs acumulada (só atualizações reais, sem zeros)
q = lambda p: (lambda s: s.quantile(p))
lat_summary = (df_upd.groupby(["Strategy", "Trigger", "TrainData"])
               .agg(n_updates_total=("latency_ms", "size"), n_train=("n_train", "median"),
                    per_update_median=("latency_ms", "median"), per_update_mean=("latency_ms", "mean"),
                    per_update_std=("latency_ms", "std"), p95=("latency_ms", q(0.95)),
                    p99=("latency_ms", q(0.99)), max=("latency_ms", "max"),
                    n_sv_median=("n_sv", "median"))
               .reset_index())
cum = (df_run.groupby("Strategy")
       .agg(updates_per_run_mean=("n_updates", "mean"),
            cumulative_per_run_mean_ms=("latency_cumulative_ms", "mean"),
            campaign_cumulative_ms=("latency_cumulative_ms",
                                    lambda s: s.sum() / args.reps / len(DETECTORS))))
lat_summary = lat_summary.merge(cum.reset_index(), on="Strategy", how="left")
lat_summary.to_csv(os.path.join(args.out_dir, "latency_summary.csv"), index=False)

# ---- Qualidade: mesma convenção de train_baseline_full.py
#      (D0 = Normal (+1); restantes cenários = Anomalia/Drift (-1)).
#      ATENÇÃO: com esta convenção, uma adaptação que aceita o novo regime
#      como normal é penalizada por construção (ver nota no artigo).
df_pred["y_true"] = np.where(df_pred["Scenario"] == "D0", 1, -1)
qual = []
for (det, strat), g in df_pred.groupby(["Detector", "Strategy"]):
    p, r, f, s = precision_recall_fscore_support(g.y_true, g.y_pred, labels=[-1, 1], zero_division=0)
    qual.append({"Detector": det, "Strategy": strat, "Scope": "ALL",
                 "F1_weighted": f1_score(g.y_true, g.y_pred, average="weighted", zero_division=0),
                 "F1_anom": f[0], "Recall_anom": r[0], "Precision_anom": p[0],
                 "Specificity": r[1], "support_anom": int(s[0]), "support_norm": int(s[1])})
    for scen, gs in g.groupby("Scenario"):
        qual.append({"Detector": det, "Strategy": strat, "Scope": scen,
                     "flag_rate": float((gs.y_pred == -1).mean()),
                     "flag_rate_after_first_update": float(
                         (gs[gs.model_version > 0].y_pred == -1).mean())
                     if (gs.model_version > 0).any() else np.nan,
                     "n": len(gs)})
pd.DataFrame(qual).to_csv(os.path.join(args.out_dir, "quality.csv"), index=False)

meta = {"finished": datetime.now().isoformat(timespec="seconds"), "elapsed_s": elapsed,
        "platform": platform.platform(), "python": sys.version.split()[0],
        "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "baseline_model": type(base_model).__name__, "n_hist": len(X_hist),
        "scenarios": scenario_files, "args": vars(args), "config_file": cfg_path,
        "strategies": {k: list(v) if v[0] else None for k, v in STRATEGIES.items()}}
with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

pd.set_option("display.width", 220)
print("\nLATÊNCIA (por atualização vs acumulada):")
print(lat_summary.round(3).to_string(index=False))
print("\nQUALIDADE (global):")
qa = pd.DataFrame(qual)
print(qa[qa.Scope == "ALL"].drop(columns=["Scope"]).dropna(axis=1, how="all").round(3).to_string(index=False))
print(f"\nResultados em {args.out_dir}")
