#!/usr/bin/env python3
"""
Descrição: Análise rigorosa das medições energéticas (FNIRSI FNB58).
Autores: Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães

Responde às revisões P0 "Clarificação da metodologia energética",
P0 "Verificação dos valores de energia" e P1 "Comparação energética
normalizada".

Dois modos:

  legacy   Reanalisa os logs existentes (data/driftsense_dataset/energy/
           energia_final_A*.csv). Reporta duração real (timestamps), taxa de
           amostragem efetiva, falhas, potência média/percentis, segmentação
           ativo/repouso, energia total, energia da fase ativa, energia
           incremental acima do repouso, e energia por stream/atualização
           (cruzando com results/.../a*run.csv).

  markers  Analisa uma campanha feita com energy_benchmark.py: cruza o log do
           medidor com markers.csv (mesmo relógio, ambos no Pi) e calcula,
           por bloco, potência média, energia incremental acima do repouso
           adjacente e energia por operação; agrega por condição (média, DP,
           IC95%) e projeta o custo por hora de operação.

Integração: o FNB58 envia pacotes USB com 4 amostras; o logger atribui o
mesmo timestamp às 4. A energia é integrada como sum(P_i * 1/fs) com
fs = 100 Hz (taxa nominal do FNB58, verificada pelos timestamps), e
validada por integração trapezoidal sobre o tempo de parede.

    python analyze_energy.py legacy
    python analyze_energy.py markers --meter power.csv --markers markers.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
ENERGY_DIR = os.path.join(PROJECT_ROOT, "data/driftsense_dataset/energy")
RUNS_DIR = os.path.join(PROJECT_ROOT, "results/driftsense_dataset/metrics")
FS_NOMINAL = 100.0


def load_meter(path):
    d = pd.read_csv(path)
    d["t"] = d["timestamp_unix"] - d["timestamp_unix"].iloc[0]
    return d


def sample_quality(d):
    t = d["timestamp_unix"].values
    dur = t[-1] - t[0]
    dt = np.diff(t)
    return {"n_samples": len(d), "wall_duration_s": dur,
            "effective_fs_hz": len(d) / dur if dur > 0 else np.nan,
            "max_gap_s": float(dt.max()) if len(dt) else np.nan,
            "n_gaps_gt_100ms": int((dt > 0.1).sum()),
            "voltage_mean_v": d["voltage_v"].mean(), "voltage_std_v": d["voltage_v"].std()}


def energy(p, t=None):
    """(E por contagem de amostras, E trapezoidal sobre o tempo de parede)."""
    e_count = float(np.sum(p) / FS_NOMINAL)
    e_trap = float(np.trapezoid(p, t)) if t is not None and len(p) > 1 else np.nan
    return e_count, e_trap


# =================================================================== LEGACY
def run_legacy(a):
    rows = []
    traces = {}
    for strat in ["A0", "A1", "A2"]:
        f = os.path.join(a.energy_dir, f"energia_final_{strat}.csv")
        d = load_meter(f)
        p, t = d["power_w"].values, d["t"].values
        r = {"Strategy": strat, "file": os.path.basename(f),
             "start": d["timestamp_iso"].iloc[0], "end": d["timestamp_iso"].iloc[-1],
             **sample_quality(d)}
        r["E_total_J"], r["E_total_trapz_J"] = energy(p, t)
        r["E_logged_J"] = float(d["energy_ws"].iloc[-1])
        r["P_mean_w"], r["P_median_w"] = p.mean(), np.median(p)
        r["P_p5_w"], r["P_p95_w"], r["P_max_w"] = np.percentile(p, 5), np.percentile(p, 95), p.max()

        # Segmentação: média móvel de 1 s acima do limiar = execução ativa
        roll = pd.Series(p).rolling(int(FS_NOMINAL), center=True, min_periods=1).mean().values
        act = np.where(roll > a.active_threshold)[0]
        i0, i1 = act[0], act[-1]
        pa = p[i0:i1 + 1]
        r["active_start_s"], r["active_end_s"] = t[i0], t[i1]
        r["active_duration_s"] = t[i1] - t[i0]
        r["P_active_mean_w"] = pa.mean()
        r["E_active_J"], _ = energy(pa)
        idle = np.r_[p[:i0], p[i1 + 1:]]
        r["idle_samples"] = len(idle)
        r["P_idle_local_w"] = idle.mean() if len(idle) >= FS_NOMINAL * 5 else np.nan
        r["truncated_at_end"] = bool((t[-1] - t[i1]) < 1.0)

        run_csv = os.path.join(a.runs_dir, f"{strat.lower()}run.csv")
        if os.path.exists(run_csv):
            rr = pd.read_csv(run_csv)
            lat = pd.to_numeric(rr["Latency (ms)"], errors="coerce").fillna(0)
            r["n_streams"] = len(rr)
            if strat == "A1":
                n_up = 2 * len(rr)  # i % 50 == 0 em 119 janelas -> i = 50, 100
            elif strat == "A2":
                n_up = int((lat > 0).sum())
            else:
                n_up = 0
            r["n_updates"] = n_up
            r["update_time_total_s"] = lat.sum() / 1000.0
        rows.append(r)
        traces[strat] = (t, p, roll)

    df = pd.DataFrame(rows).set_index("Strategy")
    # Repouso de referência: maior segmento de repouso disponível (cauda de A0)
    p_idle = a.idle_power if a.idle_power else df["P_idle_local_w"].dropna().iloc[0]
    df["P_idle_ref_w"] = p_idle
    df["E_incremental_active_J"] = (df["P_active_mean_w"] - p_idle) * df["active_duration_s"]
    df["E_active_per_stream_J"] = df["E_active_J"] / df["n_streams"]
    base = df.loc["A0"]
    df["dE_active_vs_A0_J"] = df["E_active_J"] - base["E_active_J"]
    df["dT_active_vs_A0_s"] = df["active_duration_s"] - base["active_duration_s"]
    df["dE_per_update_vs_A0_J"] = df["dE_active_vs_A0_J"] / df["n_updates"].replace(0, np.nan)
    # Estimativa pelo tempo cronometrado das atualizações (limite inferior):
    df["E_updates_from_latency_J"] = df["update_time_total_s"] * (df["P_active_mean_w"] - p_idle)
    df["unexplained_dT_s"] = df["dT_active_vs_A0_s"] - df["update_time_total_s"]

    out = os.path.join(a.out_dir, "legacy")
    os.makedirs(out, exist_ok=True)
    df.reset_index().to_csv(os.path.join(out, "energy_legacy_summary.csv"), index=False)

    tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r"Strategy & Duration (s) & Active (s) & $\bar P$ (W) & $\bar P_{act}$ (W) & $E$ (J) & $E_{act}$ (J) & $E_{act}^{inc}$ (J) \\",
           r"\midrule"]
    for s, r in df.iterrows():
        tex.append(f"{s} & {r.wall_duration_s:.1f} & {r.active_duration_s:.1f} & {r.P_mean_w:.2f} & "
                   f"{r.P_active_mean_w:.2f} & {r.E_total_J:.1f} & {r.E_active_J:.1f} & {r.E_incremental_active_J:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            f"% Idle reference power: {p_idle:.2f} W; fs = {FS_NOMINAL:.0f} Hz; inc = above idle"]
    with open(os.path.join(out, "energy_legacy_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3))
        for s, (t, p, roll) in traces.items():
            ax.plot(t, roll, label=f"{s}  ({df.loc[s, 'E_total_J']:.0f} J, {df.loc[s, 'wall_duration_s']:.0f} s)")
        ax.axhline(p_idle, ls="--", c="gray", lw=0.8, label=f"idle ≈ {p_idle:.2f} W")
        ax.set_xlabel("Time since logger start (s)"); ax.set_ylabel("Power, 1 s mean (W)")
        ax.legend(fontsize=7, frameon=False); fig.tight_layout()
        fig.savefig(os.path.join(out, "power_traces_legacy.png"), dpi=200)
        fig.savefig(os.path.join(out, "power_traces_legacy.pdf"))
    except ImportError:
        pass

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    cols = ["wall_duration_s", "effective_fs_hz", "n_gaps_gt_100ms", "P_mean_w", "active_duration_s",
            "P_active_mean_w", "P_idle_ref_w", "E_logged_J", "E_total_J", "E_total_trapz_J", "E_active_J",
            "E_incremental_active_J", "n_streams", "n_updates", "update_time_total_s",
            "dT_active_vs_A0_s", "unexplained_dT_s", "dE_per_update_vs_A0_J", "E_updates_from_latency_J",
            "truncated_at_end"]
    print(df[cols].T.round(3).to_string())
    print(f"\nResultados em {out}")


# ================================================================== MARKERS
def run_markers(a):
    d = load_meter(a.meter)
    m = pd.read_csv(a.markers)
    t_abs, p = d["timestamp_unix"].values, d["power_w"].values
    q = sample_quality(d)
    trim = a.trim

    blocks = []
    for _, b in m.iterrows():
        sel = (t_abs >= b.t_start + trim) & (t_abs <= b.t_end - trim)
        if sel.sum() < FS_NOMINAL:
            continue
        dur = (b.t_end - trim) - (b.t_start + trim)
        blocks.append({**b.to_dict(), "n_meter_samples": int(sel.sum()),
                       "dur_used_s": dur, "P_mean_w": p[sel].mean(), "P_std_w": p[sel].std()})
    bl = pd.DataFrame(blocks)

    # Repouso de referência local: média dos blocos IDLE imediatamente antes e depois
    idle = bl[bl.label == "IDLE"].reset_index(drop=True)
    work = bl[bl.label != "IDLE"].copy()

    def local_idle(ts):
        before = idle[idle.t_end <= ts].tail(1)
        after = idle[idle.t_start >= ts].head(1)
        return pd.concat([before, after]).P_mean_w.mean()

    work["P_idle_local_w"] = work.t_start.apply(local_idle)
    work["P_incr_w"] = work.P_mean_w - work.P_idle_local_w
    work["E_block_J"] = work.P_mean_w * work.dur_used_s
    work["E_incr_block_J"] = work.P_incr_w * work.dur_used_s
    ops_used = work.n_ops * work.dur_used_s / (work.t_end - work.t_start)
    work["E_incr_per_op_mJ"] = 1e3 * work.E_incr_block_J / ops_used
    work["time_per_op_ms"] = 1e3 * (work.t_end - work.t_start) / work.n_ops

    def ci95(s):
        s = s.dropna()
        if len(s) < 2:
            return np.nan
        return stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))

    agg = work.groupby("label").agg(
        rounds=("P_mean_w", "size"), P_mean_w=("P_mean_w", "mean"),
        P_idle_w=("P_idle_local_w", "mean"), P_incr_w=("P_incr_w", "mean"),
        P_incr_ci95=("P_incr_w", ci95), time_per_op_ms=("time_per_op_ms", "mean"),
        E_incr_per_op_mJ=("E_incr_per_op_mJ", "mean"),
        E_incr_per_op_sd=("E_incr_per_op_mJ", "std"),
        E_incr_per_op_ci95=("E_incr_per_op_mJ", ci95)).reset_index()

    # Projeção por hora de operação a 1 janela / 5 s (720 janelas/h):
    # A0 = 720 inferências; A1 = 720 inf. + 3600/(5*50) = 14.4 atualizações;
    # A2 = 720 inf. + k atualizações por evento de drift (k = --a2-updates-per-hour).
    e = agg.set_index("label").E_incr_per_op_mJ
    p_idle = idle.P_mean_w.mean()
    proj = []
    infer = e.get("INFER", np.nan)
    for name, upd_label, n_upd in [("A0", None, 0.0),
                                   ("A1", "A1_HIST139_mem", 3600 / (5 * a.a1_interval)),
                                   ("A1 (orig., disk I/O)", "A1_HIST139_disk", 3600 / (5 * a.a1_interval)),
                                   ("A2", "A2_BUF20", a.a2_updates_per_hour)]:
        e_upd = e.get(upd_label, np.nan) if upd_label else 0.0
        e_inc = (720 * infer + n_upd * e_upd) / 1e3
        proj.append({"Strategy": name, "updates_per_h": n_upd, "E_incr_per_h_J": e_inc,
                     "E_idle_per_h_J": p_idle * 3600, "E_total_per_h_J": p_idle * 3600 + e_inc,
                     "incremental_share_pct": 100 * e_inc / (p_idle * 3600 + e_inc)})
    proj = pd.DataFrame(proj)

    out = os.path.join(a.out_dir, "markers")
    os.makedirs(out, exist_ok=True)
    work.to_csv(os.path.join(out, "energy_blocks.csv"), index=False)
    agg.to_csv(os.path.join(out, "energy_per_condition.csv"), index=False)
    proj.to_csv(os.path.join(out, "energy_projection_per_hour.csv"), index=False)
    pd.Series(q).to_csv(os.path.join(out, "meter_quality.csv"))

    pd.set_option("display.width", 250)
    print("Qualidade do log:", {k: round(v, 3) if isinstance(v, float) else v for k, v in q.items()})
    print(f"Repouso médio: {p_idle:.3f} W (DP entre blocos {idle.P_mean_w.std():.3f} W, n={len(idle)})\n")
    print(agg.round(4).to_string(index=False))
    print("\nProjeção por hora (720 janelas/h):")
    print(proj.round(3).to_string(index=False))
    print(f"\nResultados em {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "energy_analysis"))
    sub = ap.add_subparsers(dest="mode", required=True)
    lg = sub.add_parser("legacy")
    lg.add_argument("--energy-dir", default=ENERGY_DIR)
    lg.add_argument("--runs-dir", default=RUNS_DIR)
    lg.add_argument("--active-threshold", type=float, default=4.5, help="W (média móvel 1 s)")
    lg.add_argument("--idle-power", type=float, default=None, help="Forçar potência de repouso (W)")
    mk = sub.add_parser("markers")
    mk.add_argument("--meter", required=True, help="CSV do power_meter_fnirsi.py")
    mk.add_argument("--markers", required=True, help="markers.csv do energy_benchmark.py")
    mk.add_argument("--trim", type=float, default=2.0, help="s descartados no início/fim de cada bloco")
    mk.add_argument("--a1-interval", type=int, default=50)
    mk.add_argument("--a2-updates-per-hour", type=float, default=1.0)
    a = ap.parse_args()
    run_legacy(a) if a.mode == "legacy" else run_markers(a)
