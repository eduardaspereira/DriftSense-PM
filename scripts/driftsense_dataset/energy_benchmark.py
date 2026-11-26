#!/usr/bin/env python3
"""
Descrição: Campanha energética controlada com marcadores temporais, para
           medir a energia INCREMENTAL por operação (inferência e atualização).
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Responde à revisão P0 "Clarificação da metodologia energética": separa a
energia total do sistema da energia incremental das operações de adaptação.

Protocolo (blocos de duração fixa -> comparações com a mesma duração):
  1. IDLE inicial (--idle-first s): processo vivo, a dormir -> potência de repouso
  2. R rondas; em cada ronda, ordem aleatória das condições:
        bloco de trabalho (--block s) a repetir a operação em ciclo
        IDLE (--idle s) entre blocos -> repouso local, antes e depois
  3. markers.csv: label, round, t_start, t_end, n_ops  (time.time(), relógio do Pi)

Condições (as mesmas operações de adaptations.py):
  INFER            scaler.transform + OC-SVM.predict de 1 vetor (custo A0 por janela)
  A2_BUF20         refit scaler + OC-SVM com 20 vetores (A2 original)
  A2_BUF139        refit com 139 vetores recentes (N igual ao de A1)
  A1_HIST139_mem   refit com D0 (119) + 20 recentes, D0 já em memória
  A1_HIST139_disk  igual, mas lê D0 do disco a cada atualização (A1 original)

Execução no Raspberry Pi 5 (o medidor e este script no MESMO Pi):
  terminal 1:  python power_meter_fnirsi.py -o energy_campaign_meter.csv -d 3000 -i 1
  terminal 2:  python energy_benchmark.py --rounds 5 --block 30 --idle 20
  depois:      python analyze_energy.py markers --meter energy_campaign_meter.csv \
                   --markers ../../results/driftsense_dataset/metrics/energy_campaign/markers.csv
Duração com os valores por omissão: ~60 + 5*5*(30+20) s ≈ 22 min.
Ver também o checklist no docstring de analyze_energy.py e no README da pasta.
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
META_COLS = ['Scenario', 'Timestamp', 'SysState', 'SampleCount']

ap = argparse.ArgumentParser()
ap.add_argument("--processed-dir", default=os.path.join(PROJECT_ROOT, "data/driftsense_dataset/processed"))
ap.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models/driftsense_dataset"))
ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results/driftsense_dataset/metrics/energy_campaign"))
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--block", type=float, default=30.0, help="s de trabalho por bloco")
ap.add_argument("--idle", type=float, default=20.0, help="s de repouso entre blocos")
ap.add_argument("--idle-first", type=float, default=60.0)
ap.add_argument("--cpu", type=int, default=None, help="Fixar num core")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--conditions", default="INFER,A2_BUF20,A2_BUF139,A1_HIST139_mem,A1_HIST139_disk")
a = ap.parse_args()

if a.cpu is not None and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {a.cpu})
random.seed(a.seed)
rng = np.random.default_rng(a.seed)

d0_file = [f for f in os.listdir(a.processed_dir) if f.startswith('D0_')][0]
X_d0 = pd.read_csv(os.path.join(a.processed_dir, d0_file)).drop(META_COLS, axis=1, errors='ignore')
cols = list(X_d0.columns)
pool = pd.concat([pd.read_csv(os.path.join(a.processed_dir, f)).drop(META_COLS, axis=1, errors='ignore')[cols]
                  for f in sorted(os.listdir(a.processed_dir))
                  if f.endswith('.csv') and f != d0_file and not f.startswith('D5')], ignore_index=True)
model0 = joblib.load(os.path.join(a.models_dir, 'baseline_model.pkl'))
scaler0 = joblib.load(os.path.join(a.models_dir, 'scaler.pkl'))


def recent(n):
    s = int(rng.integers(0, len(pool) - n + 1))
    return pool.iloc[s:s + n]


def refit(X):
    sc = StandardScaler()
    OneClassSVM(kernel='rbf', nu=0.05, gamma='auto').fit(sc.fit_transform(X))


k_inf = [0]


def op_infer():
    i = k_inf[0] % len(pool)
    k_inf[0] += 1
    model0.predict(scaler0.transform(pool.iloc[[i]]))


def op_a1(disk):
    def f():
        if disk:
            fn = [x for x in os.listdir(a.processed_dir) if x.startswith('D0_')][0]
            hist = pd.read_csv(os.path.join(a.processed_dir, fn)).drop(META_COLS, axis=1, errors='ignore')
        else:
            hist = X_d0
        refit(pd.concat([hist, pd.DataFrame(recent(20).values, columns=hist.columns)], ignore_index=True))
    return f


OPS = {"INFER": op_infer,
       "A2_BUF20": lambda: refit(recent(20)),
       "A2_BUF139": lambda: refit(recent(139)),
       "A1_HIST139_mem": op_a1(False),
       "A1_HIST139_disk": op_a1(True)}
conds = [c.strip() for c in a.conditions.split(",")]

os.makedirs(a.out_dir, exist_ok=True)
markers = []


def temp():
    try:
        return float(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        return None


def idle_block(r, dur):
    t0 = time.time()
    time.sleep(dur)
    markers.append({"label": "IDLE", "round": r, "t_start": t0, "t_end": time.time(),
                    "n_ops": 0, "temp_c": temp()})


def work_block(label, r):
    f = OPS[label]
    for _ in range(3):  # aquecimento fora do bloco medido
        f()
    n, t0 = 0, time.time()
    while time.time() - t0 < a.block:
        f()
        n += 1
    markers.append({"label": label, "round": r, "t_start": t0, "t_end": time.time(),
                    "n_ops": n, "temp_c": temp()})
    print(f"  ronda {r} | {label:<16} {n:>7} ops  ({1e3 * a.block / max(n, 1):.3f} ms/op)  T={temp()} C")


print(f"IDLE inicial {a.idle_first:.0f} s  (confirma que o medidor já está a gravar!)")
idle_block(0, a.idle_first)
for r in range(1, a.rounds + 1):
    order = conds[:]
    random.shuffle(order)
    for c in order:
        work_block(c, r)
        idle_block(r, a.idle)
    pd.DataFrame(markers).to_csv(os.path.join(a.out_dir, "markers.csv"), index=False)  # guarda progressivo

pd.DataFrame(markers).to_csv(os.path.join(a.out_dir, "markers.csv"), index=False)
meta = {"finished": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
        "python": sys.version.split()[0], "sklearn": sklearn.__version__, "args": vars(a),
        "governor": open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
        if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") else None,
        "baseline_model": type(model0).__name__, "n_d0": len(X_d0)}
with open(os.path.join(a.out_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nFeito. Marcadores em {a.out_dir}/markers.csv — já podes parar o medidor (Ctrl+C).")
