import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features import airport_delay_aggregates


def run_isolation_forest_airports(df_work, out_dir: Path, random_state=42, contamination=0.02, min_flights=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    g = airport_delay_aggregates(df_work)
    if min_flights is None:
        min_flights = max(8, min(200, int(len(df_work) * 0.008)))
    g = g.loc[g["n_flights"] >= min_flights].copy()
    while len(g) < 12 and min_flights > 1:
        min_flights = max(1, min_flights - 1)
        g = airport_delay_aggregates(df_work)
        g = g.loc[g["n_flights"] >= min_flights].copy()
    feat_cols = ["n_flights", "mean_dep_delay", "median_dep_delay", "pct_gt15"]
    X = g[feat_cols].values
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(random_state=random_state, contamination=contamination, n_estimators=300)
    g["anomaly_score"] = iso.fit_predict(Xs)
    g["anomaly_decision"] = g["anomaly_score"].map({1: "normal", -1: "anomalo"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = g.copy()
    pal = {"normal": "steelblue", "anomalo": "crimson"}
    sns.scatterplot(
        data=sub,
        x="mean_dep_delay",
        y="pct_gt15",
        hue="anomaly_decision",
        palette=pal,
        size="n_flights",
        sizes=(25, 350),
        ax=ax,
        alpha=0.85,
    )
    ax.set_title("Deteccao de anomalias em aeroportos (Isolation Forest)")
    ax.set_xlabel("Atraso medio na partida (min)")
    ax.set_ylabel("Proporcao atraso > 15 min")
    fig.tight_layout()
    fig.savefig(out_dir / "anomaly_airports.png", dpi=150)
    plt.close(fig)
    g.to_csv(out_dir / "airport_anomalies.csv", index=False)
    n_anom = int((g["anomaly_score"] == -1).sum())
    summary = {"n_airports": int(len(g)), "n_flagged": n_anom, "contamination": contamination}
    (out_dir / "anomaly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
