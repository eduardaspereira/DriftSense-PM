#!/usr/bin/env python3
"""
Descrição: Recalcula as métricas da Tabela III diretamente a partir das
           previsões, com matrizes de confusão, suporte por classe e
           intervalos de confiança.
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Responde à revisão P0 "Verificação das métricas da Tabela III".

Replica exatamente o protocolo de train_baseline_full.py:
  * D0 (normal, +1) dividido cronologicamente 80/20 (shuffle=False)
  * teste = 20% final de D0 + TODAS as janelas dos cenários D1..D4 (-1)
  * StandardScaler ajustado só ao treino de D0
Para o OC-SVM avalia a configuração da Tabela III (nu=0.01, gamma='scale',
os valores por omissão de train_baseline_full.py) e a usada nas adaptações
(nu=0.05, gamma='auto'), já que o artigo menciona ambas.

Saídas (results/driftsense_dataset/metrics/table3_recomputed/):
  predictions.csv, confusion_matrices.csv, metrics.csv, per_scenario.csv,
  table3.tex
Determinístico, não precisa do Pi:
    python recompute_baseline_metrics.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
META_COLS = ['Scenario', 'Timestamp', 'SysState', 'SampleCount']

parser = argparse.ArgumentParser()
parser.add_argument("--processed-dir", default=os.path.join(PROJECT_ROOT, "data/driftsense_dataset/processed"))
parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results/driftsense_dataset/metrics/table3_recomputed"))
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

# ---------------- dados (idêntico a train_baseline_full.py)
df_d0 = pd.read_csv(os.path.join(args.processed_dir, "D0_dataset_features.csv"))
X_d0 = df_d0.drop(META_COLS, axis=1, errors='ignore')
X_train, X_test_n = train_test_split(X_d0, test_size=0.2, shuffle=False)

parts = [X_test_n.assign(_scen="D0")]
for f in sorted(os.listdir(args.processed_dir)):          # sorted -> ordem reprodutível
    if f.endswith(".csv") and not f.startswith("D0"):
        d = pd.read_csv(os.path.join(args.processed_dir, f)).drop(META_COLS, axis=1, errors='ignore')
        parts.append(d.assign(_scen=f.split("_dataset")[0]))
test = pd.concat(parts, ignore_index=True)
scen = test.pop("_scen").values
X_test = test[X_train.columns]
y_test = np.where(scen == "D0", 1, -1)

scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

MODELS = {   # hiperparâmetros = defaults de train_baseline_full.py
    "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.1 / 100, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=1.0 / 100, novelty=True),
    "One-Class SVM (nu=0.01, gamma=scale)": OneClassSVM(kernel="rbf", nu=0.01, gamma="scale"),
    "One-Class SVM (nu=0.05, gamma=auto)": OneClassSVM(kernel="rbf", nu=0.05, gamma="auto"),
}


def cp_interval(k, n, a=0.05):
    """Intervalo exato de Clopper-Pearson para uma proporção k/n."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def metrics_from(y, p):
    cm = confusion_matrix(y, p, labels=[-1, 1])          # linhas = verdade, colunas = previsto
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    n_a, n_n = tp + fn, fp + tn
    prec_a = tp / (tp + fp) if tp + fp else 0.0
    rec_a = tp / n_a if n_a else 0.0
    f1_a = 2 * prec_a * rec_a / (prec_a + rec_a) if prec_a + rec_a else 0.0
    prec_n = tn / (tn + fn) if tn + fn else 0.0
    rec_n = tn / n_n if n_n else 0.0                     # = especificidade
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if prec_n + rec_n else 0.0
    return dict(TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn),
                support_anom=int(n_a), support_norm=int(n_n),
                precision_anom=prec_a, recall_anom=rec_a, f1_anom=f1_a,
                precision_norm=prec_n, specificity=rec_n, f1_norm=f1_n, fpr=1 - rec_n,
                macro_f1=(f1_a + f1_n) / 2,
                weighted_f1=(n_a * f1_a + n_n * f1_n) / (n_a + n_n),
                balanced_acc=(rec_a + rec_n) / 2,
                mcc=matthews_corrcoef(y, p),
                accuracy=(tp + tn) / (n_a + n_n))


def bootstrap_ci(y, p, keys=("macro_f1", "weighted_f1", "balanced_acc", "mcc")):
    """Bootstrap estratificado (reamostra cada classe separadamente)."""
    ia, iN = np.where(y == -1)[0], np.where(y == 1)[0]
    acc = {k: [] for k in keys}
    for _ in range(args.bootstrap):
        idx = np.r_[rng.choice(ia, len(ia)), rng.choice(iN, len(iN))]
        m = metrics_from(y[idx], p[idx])
        for k in keys:
            acc[k].append(m[k])
    return {f"{k}_ci95": (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in acc.items()}


os.makedirs(args.out_dir, exist_ok=True)
rows, cms, per_scen = [], [], []
pred_df = pd.DataFrame({"Scenario": scen, "y_true": y_test})
for name, model in MODELS.items():
    model.fit(Xtr)
    p = model.predict(Xte).astype(int)
    pred_df[name] = p
    m = metrics_from(y_test, p)
    m["specificity_ci95"] = cp_interval(m["TN"], m["support_norm"])
    m["recall_anom_ci95"] = cp_interval(m["TP"], m["support_anom"])
    m.update(bootstrap_ci(y_test, p))
    rows.append({"Model": name, **m})
    cms.append({"Model": name, "true=-1,pred=-1 (TP)": m["TP"], "true=-1,pred=+1 (FN)": m["FN"],
                "true=+1,pred=-1 (FP)": m["FP"], "true=+1,pred=+1 (TN)": m["TN"]})
    for s in pd.unique(scen):
        mask = scen == s
        flagged = int((p[mask] == -1).sum())
        lo, hi = cp_interval(flagged, int(mask.sum()))
        per_scen.append({"Model": name, "Scenario": s, "n": int(mask.sum()),
                         "true_class": "Normal(+1)" if s == "D0" else "Anomaly(-1)",
                         "flagged_anomalous": flagged,
                         "rate": flagged / mask.sum(),
                         "metric": "FPR" if s == "D0" else "Recall_anom",
                         "ci95_lo": lo, "ci95_hi": hi})

met = pd.DataFrame(rows)
pred_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
pd.DataFrame(cms).to_csv(os.path.join(args.out_dir, "confusion_matrices.csv"), index=False)
met.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
pd.DataFrame(per_scen).to_csv(os.path.join(args.out_dir, "per_scenario.csv"), index=False)

# ---------------- tabela LaTeX pronta a colar
fmt = lambda x: f"{x:.3f}"
ci = lambda t: f"[{t[0]:.2f}, {t[1]:.2f}]"
lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"Model & TP/FN/FP/TN & Prec.$_a$ & Rec.$_a$ & F1$_a$ & Spec. [95\% CI] & Bal. Acc. & Macro-F1 & MCC \\",
         r"\midrule"]
for _, r in met.iterrows():
    lines.append(f"{r.Model} & {r.TP}/{r.FN}/{r.FP}/{r.TN} & {fmt(r.precision_anom)} & {fmt(r.recall_anom)} & "
                 f"{fmt(r.f1_anom)} & {fmt(r.specificity)} {ci(r.specificity_ci95)} & {fmt(r.balanced_acc)} & "
                 f"{fmt(r.macro_f1)} & {fmt(r.mcc)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          f"% Support: {met.support_anom.iloc[0]} anomalous (D1--D4) / {met.support_norm.iloc[0]} normal (last 20\\% of D0)"]
with open(os.path.join(args.out_dir, "table3.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

pd.set_option("display.width", 250)
print(f"Treino: {len(X_train)} janelas D0 | Teste: {int((y_test == -1).sum())} anómalas + {int((y_test == 1).sum())} normais")
print(met[["Model", "TP", "FN", "FP", "TN", "precision_anom", "recall_anom", "f1_anom", "specificity",
           "specificity_ci95", "macro_f1", "weighted_f1", "balanced_acc", "mcc"]].round(3).to_string(index=False))
print("\nPor cenário:")
print(pd.DataFrame(per_scen).round(3).to_string(index=False))
print(f"\nResultados em {args.out_dir}")
