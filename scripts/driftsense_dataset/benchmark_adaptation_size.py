#!/usr/bin/env python3
"""
Descrição: Micro-benchmark do custo de UMA atualização do modelo (A1 vs A2)
           com conjuntos de treino de dimensão equivalente.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Responde à revisão P1 "Comparabilidade das estratégias A1 e A2": no desenho
original, A1 treina com D0 completo + buffer (119 + 20 = 139 vetores) e lê o
CSV de D0 do disco dentro do cronómetro, enquanto A2 treina só com o buffer
(20 vetores). Este script separa os três fatores:

  * composição dos dados (mecanismo):  HIST+BUF (A1)  vs  BUF (A2)
  * dimensão do conjunto de treino N:   p.ex. 20, 40, 70, 100, 139
  * I/O de disco dentro do cronómetro:  memory  vs  disk  (só HIST+BUF,
                                         que é o único que lê D0 do disco)

A operação cronometrada é exatamente a de adaptations.py:
    StandardScaler().fit_transform(X) + OneClassSVM(rbf, nu=0.05, gamma='auto').fit
São registadas as fases (io / scaler / fit) separadamente.

Não requer hardware nem novos cenários: usa apenas os CSV em
data/driftsense_dataset/processed/. Deve ser corrido no Raspberry Pi 5.

Uso (no Pi, com o ventilador/ambiente estáveis e sem outros processos):
    python benchmark_adaptation_size.py --reps 200 --cpu 3
"""

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_PROCESSED = os.path.join(PROJECT_ROOT, "data/driftsense_dataset/processed")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "results/driftsense_dataset/metrics/adaptation_benchmark")
META_COLS = ['Scenario', 'Timestamp', 'SysState', 'SampleCount']

parser = argparse.ArgumentParser(description="Benchmark A1 vs A2 com N equivalente")
parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED)
parser.add_argument("--out-dir", default=DEFAULT_OUT)
parser.add_argument("--sizes", default="20,40,70,100,139",
                    help="Dimensões N do conjunto de treino (vírgulas)")
parser.add_argument("--buffer-size", type=int, default=20,
                    help="Dimensão do buffer recente usado por A1 (HIST+BUF)")
parser.add_argument("--reps", type=int, default=200, help="Repetições por condição")
parser.add_argument("--warmup", type=int, default=10, help="Execuções de aquecimento por condição")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cpu", type=int, default=None,
                    help="Fixar o processo num core (ex.: 3) para reduzir variância")
parser.add_argument("--nu", type=float, default=0.05)
parser.add_argument("--gamma", default="auto")
args = parser.parse_args()


# ---------------------------------------------------------------- utilitários
def read_sys(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cpu_temp_c():
    t = read_sys("/sys/class/thermal/thermal_zone0/temp")
    return float(t) / 1000.0 if t else None


def throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                              text=True, timeout=2).stdout.strip()
    except Exception:
        return None


def d0_path(processed_dir):
    return os.path.join(processed_dir,
                        [f for f in os.listdir(processed_dir) if f.startswith('D0_')][0])


def load_features(path):
    return pd.read_csv(path).drop(META_COLS, axis=1, errors='ignore')


# ------------------------------------------------------------------- dados
if args.cpu is not None and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {args.cpu})

rng = np.random.default_rng(args.seed)
random.seed(args.seed)

X_d0 = load_features(d0_path(args.processed_dir))
cols = list(X_d0.columns)
drift_files = sorted(f for f in os.listdir(args.processed_dir)
                     if f.endswith('.csv') and not f.startswith('D0_') and not f.startswith('D5'))
drift_pool = pd.concat([load_features(os.path.join(args.processed_dir, f))[cols]
                        for f in drift_files], ignore_index=True)
D0_PATH = d0_path(args.processed_dir)
N_HIST_MAX = len(X_d0)
SIZES = [int(s) for s in args.sizes.split(",")]

print(f"D0: {N_HIST_MAX} vetores | pool de drift: {len(drift_pool)} vetores ({', '.join(drift_files)})")
print(f"Features: {len(cols)} | N = {SIZES} | reps = {args.reps} | warmup = {args.warmup}")


def recent_buffer(n):
    """Fatia contígua do stream de drift (simula as n janelas mais recentes)."""
    start = int(rng.integers(0, len(drift_pool) - n + 1))
    return drift_pool.iloc[start:start + n]


def split_hist_buf(n):
    """Para HIST+BUF com N total: n_buf recentes + (N - n_buf) de D0."""
    n_buf = min(args.buffer_size, n // 2)
    n_hist = min(n - n_buf, N_HIST_MAX)
    return n_hist, n - n_hist


# ------------------------------------------------------- operação medida
def update(mechanism, n, io):
    """Executa uma atualização e devolve tempos por fase (ms) e nº de SVs."""
    if mechanism == "BUF":
        X_new = recent_buffer(n)
        t0 = time.perf_counter()
        X = X_new
        t_io = time.perf_counter()
    else:  # HIST+BUF
        n_hist, n_buf = split_hist_buf(n)
        X_new = recent_buffer(n_buf)
        hist_idx = None if n_hist == N_HIST_MAX else np.sort(
            rng.choice(N_HIST_MAX, n_hist, replace=False))
        t0 = time.perf_counter()
        if io == "disk":  # replica adaptations.apply_a1_periodic_retrain
            caminho = [f for f in os.listdir(args.processed_dir) if f.startswith('D0_')][0]
            hist = load_features(os.path.join(args.processed_dir, caminho))
        else:
            hist = X_d0
        if hist_idx is not None:
            hist = hist.iloc[hist_idx]
        X = pd.concat([hist, pd.DataFrame(X_new.values, columns=hist.columns)],
                      ignore_index=True)
        t_io = time.perf_counter()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    t_sc = time.perf_counter()
    model = OneClassSVM(kernel='rbf', nu=args.nu, gamma=args.gamma)
    model.fit(Xs)
    t_fit = time.perf_counter()
    return {
        "n_train": len(X),
        "prep_ms": (t_io - t0) * 1e3,
        "scaler_ms": (t_sc - t_io) * 1e3,
        "fit_ms": (t_fit - t_sc) * 1e3,
        "total_ms": (t_fit - t0) * 1e3,
        "n_sv": int(model.support_vectors_.shape[0]),
    }


conditions = []
for n in SIZES:
    conditions.append(("BUF", n, "memory"))
    for io in ("memory", "disk"):
        conditions.append(("HIST+BUF", n, io))

# ------------------------------------------------------------- execução
meta = {
    "started": datetime.now().isoformat(timespec="seconds"),
    "platform": platform.platform(), "machine": platform.machine(),
    "python": sys.version.split()[0], "numpy": np.__version__,
    "pandas": pd.__version__, "sklearn": sklearn.__version__,
    "cpu_model": read_sys("/proc/device-tree/model"),
    "governor": read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    "cur_freq_khz": read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
    "cpu_affinity": args.cpu, "temp_start_c": cpu_temp_c(), "throttled_start": throttled(),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "args": vars(args), "n_d0": N_HIST_MAX, "drift_files": drift_files,
    "n_features": len(cols),
}

print("Aquecimento...")
for c in conditions:
    for _ in range(args.warmup):
        update(*c)

# Ordem intercalada e aleatória: evita que deriva térmica/frequência
# favoreça sistematicamente uma condição.
schedule = [c for c in conditions for _ in range(args.reps)]
random.shuffle(schedule)

rows = []
t_start = time.perf_counter()
for k, (mech, n, io) in enumerate(schedule):
    r = update(mech, n, io)
    r.update({"trial": k, "mechanism": mech, "N_target": n, "io": io,
              "strategy_equiv": "A2" if mech == "BUF" else "A1"})
    rows.append(r)
    if k % 500 == 0:
        print(f"  {k}/{len(schedule)}  temp={cpu_temp_c()} C")
meta["bench_duration_s"] = time.perf_counter() - t_start

# Memória: passagem separada (tracemalloc distorce os tempos).
# Nota: tracemalloc vê alocações Python/NumPy; buffers internos da libsvm (C)
# não são contabilizados -> reportar como limite inferior.
mem_rows = []
for c in conditions:
    for _ in range(5):
        tracemalloc.start()
        update(*c)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_rows.append({"mechanism": c[0], "N_target": c[1], "io": c[2],
                         "peak_kib": peak / 1024})
meta["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
meta["temp_end_c"] = cpu_temp_c()
meta["throttled_end"] = throttled()

# -------------------------------------------------------------- resultados
os.makedirs(args.out_dir, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out_dir, "raw_trials.csv"), index=False)


def q(p):
    return lambda s: s.quantile(p)


summary = (df.groupby(["mechanism", "io", "N_target"])
           .agg(n_train=("n_train", "first"), reps=("total_ms", "size"),
                total_median=("total_ms", "median"), total_mean=("total_ms", "mean"),
                total_std=("total_ms", "std"), total_p5=("total_ms", q(0.05)),
                total_p95=("total_ms", q(0.95)), total_p99=("total_ms", q(0.99)),
                total_max=("total_ms", "max"),
                prep_median=("prep_ms", "median"), scaler_median=("scaler_ms", "median"),
                fit_median=("fit_ms", "median"), n_sv_median=("n_sv", "median"))
           .reset_index())
mem = pd.DataFrame(mem_rows).groupby(["mechanism", "io", "N_target"])["peak_kib"].median()
summary = summary.merge(mem.rename("tracemalloc_peak_kib").reset_index(),
                        on=["mechanism", "io", "N_target"])

# Rácio HIST+BUF / BUF com o MESMO N (efeito do mecanismo isolado)
buf = summary[summary.mechanism == "BUF"].set_index("N_target")["total_median"]
summary["ratio_vs_BUF_sameN"] = summary.apply(
    lambda r: r.total_median / buf.get(r.N_target, np.nan), axis=1)
summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

pd.set_option("display.width", 200)
print("\n" + summary[["mechanism", "io", "N_target", "n_train", "total_median", "total_p95",
                      "total_p99", "prep_median", "fit_median", "n_sv_median",
                      "tracemalloc_peak_kib", "ratio_vs_BUF_sameN"]].round(3).to_string(index=False))
print(f"\nResultados em {args.out_dir}")
print(f"Temp. CPU: {meta['temp_start_c']} -> {meta['temp_end_c']} C | throttled: {meta['throttled_end']}")
