import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def describe_and_save(df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = df.describe(include="all")
    desc.to_csv(out_dir / "describe_all.csv")
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(out_dir / "missing_rate.csv", header=["missing_rate"])
    with open(out_dir / "missing_summary.json", "w", encoding="utf-8") as f:
        json.dump(miss.head(40).to_dict(), f, ensure_ascii=False, indent=2)


def plot_eda(df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    d = df.loc[df["CANCELLED"] == 0].dropna(subset=["DEPARTURE_DELAY"]).copy()
    d["delayed_15"] = (d["DEPARTURE_DELAY"] > 15).astype(int)

    fig, ax = plt.subplots(figsize=(8, 4))
    d["DEPARTURE_DELAY"].clip(-50, 120).hist(bins=80, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Distribuicao de atraso na partida (min), recorte -50 a 120")
    ax.set_xlabel("DEPARTURE_DELAY")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_departure_delay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    by_dow = d.groupby("DAY_OF_WEEK")["delayed_15"].mean()
    by_dow.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Taxa de atraso > 15 min por dia da semana")
    ax.set_xlabel("DAY_OF_WEEK (1=Seg ... 7=Dom)")
    ax.set_ylabel("Proporcao")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_rate_by_dow.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    d["dep_h"] = (pd.to_numeric(d["SCHEDULED_DEPARTURE"], errors="coerce").fillna(0).astype(int) // 100).clip(0, 23)
    by_h = d.groupby("dep_h")["delayed_15"].mean()
    by_h.plot(kind="line", ax=ax, marker="o", color="darkorange")
    ax.set_title("Taxa de atraso > 15 min por hora programada de partida")
    ax.set_xlabel("Hora (SCHEDULED_DEPARTURE)")
    ax.set_ylabel("Proporcao")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_rate_by_hour.png", dpi=150)
    plt.close(fig)

    top_airlines = d["AIRLINE"].value_counts().nlargest(12).index
    sub = d.loc[d["AIRLINE"].isin(top_airlines)].copy()
    sub["dd_clip"] = sub["DEPARTURE_DELAY"].clip(-30, 90)
    fig, ax = plt.subplots(figsize=(9, 4))
    order = (
        sub.groupby("AIRLINE")["delayed_15"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxenplot(data=sub, x="AIRLINE", y="dd_clip", order=order, ax=ax)
    ax.set_title("Atraso na partida por companhia (top 12 volume), recorte eixo Y")
    ax.set_ylabel("DEPARTURE_DELAY")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_by_airline.png", dpi=150)
    plt.close(fig)

    top_o = d["ORIGIN_AIRPORT"].value_counts().nlargest(20).index
    subo = d.loc[d["ORIGIN_AIRPORT"].isin(top_o)]
    agg = subo.groupby("ORIGIN_AIRPORT")["delayed_15"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    agg.plot(kind="barh", ax=ax, color="slategray")
    ax.set_title("Taxa atraso > 15 min — top 20 aeroportos por volume")
    ax.set_xlabel("Proporcao")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_rate_top_origins.png", dpi=150)
    plt.close(fig)

    if "MONTH" in d.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        d.groupby("MONTH")["delayed_15"].mean().plot(kind="bar", ax=ax, color="indigo")
        ax.set_title("Sazonalidade: taxa de atraso > 15 min por mes")
        ax.set_xlabel("Mes")
        fig.tight_layout()
        fig.savefig(out_dir / "delay_rate_by_month.png", dpi=150)
        plt.close(fig)

    if "ORIGIN_STATE" in d.columns:
        st_sub = d.loc[d["ORIGIN_STATE"].astype(str) != "UNK"]
        if len(st_sub) > 100 and st_sub["ORIGIN_STATE"].nunique() > 1:
            top_st = st_sub["ORIGIN_STATE"].value_counts().nlargest(30).index
            st2 = st_sub.loc[st_sub["ORIGIN_STATE"].isin(top_st)]
            agg_st = st2.groupby("ORIGIN_STATE")["delayed_15"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 5))
            agg_st.plot(kind="barh", ax=ax, color="darkslateblue")
            ax.set_title("Taxa atraso > 15 min por estado (origem), top 30 por volume")
            ax.set_xlabel("Proporcao")
            fig.tight_layout()
            fig.savefig(out_dir / "delay_rate_by_origin_state.png", dpi=150)
            plt.close(fig)
