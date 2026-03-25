import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features import airport_delay_aggregates


def run_pca_on_flights(df_work, out_dir: Path, random_state=42, max_points=25000):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = df_work.copy()
    num_cols = [
        "DISTANCE",
        "SCHEDULED_TIME",
        "dep_hour",
        "MONTH",
        "DAY_OF_WEEK",
        "DEPARTURE_DELAY",
    ]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=num_cols)
    if len(d) > max_points:
        d = d.sample(max_points, random_state=random_state)
    X = d[num_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_.tolist()
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=d["DEPARTURE_DELAY"].clip(-30, 90), cmap="viridis", alpha=0.35, s=10)
    ax.set_title("PCA (2D) colorido por atraso na partida")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    fig.colorbar(sc, ax=ax, label="DEPARTURE_DELAY (clip)")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_flights_2d.png", dpi=150)
    plt.close(fig)
    payload = {"explained_variance_ratio": [float(x) for x in evr], "n_samples": int(len(d))}
    (out_dir / "pca_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_airport_clustering(df_work, out_dir: Path, random_state=42, min_flights=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    g = airport_delay_aggregates(df_work)
    if min_flights is None:
        min_flights = max(8, min(200, int(len(df_work) * 0.008)))
    g = g.loc[g["n_flights"] >= min_flights]
    while len(g) < 6 and min_flights > 1:
        min_flights = max(1, min_flights - 1)
        g = airport_delay_aggregates(df_work)
        g = g.loc[g["n_flights"] >= min_flights]
    feat_cols = ["n_flights", "mean_dep_delay", "median_dep_delay", "pct_gt15"]
    X = g[feat_cols].values
    X = StandardScaler().fit_transform(X)
    best_k = None
    best_score = -1.0
    record = {}
    max_k = min(8, max(2, len(g) - 1))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        record[str(k)] = sil
        if sil > best_score:
            best_score = sil
            best_k = k
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    g = g.copy()
    g["cluster"] = labels
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=g,
        x="mean_dep_delay",
        y="pct_gt15",
        hue="cluster",
        size="n_flights",
        sizes=(20, 400),
        palette="tab10",
        ax=ax,
        alpha=0.85,
    )
    ax.set_title(f"Clusters de aeroportos (K={best_k}) perfil de atraso")
    ax.set_xlabel("Atraso medio na partida (min)")
    ax.set_ylabel("Proporcao voos com atraso > 15 min")
    fig.tight_layout()
    fig.savefig(out_dir / "clusters_airports_scatter.png", dpi=150)
    plt.close(fig)
    g.to_csv(out_dir / "airport_clusters.csv", index=False)
    summary = {"best_k": int(best_k), "silhouette_by_k": record, "silhouette_best": float(best_score)}
    (out_dir / "clustering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, g
