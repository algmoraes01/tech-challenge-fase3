import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs"


def load_json(p):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def show_dataframe(df):
    try:
        st.dataframe(df, width="stretch")
    except (TypeError, ValueError):
        st.dataframe(df, use_container_width=True)


def main():
    st.set_page_config(page_title="Voos — Tech Challenge Fase 3", layout="wide")
    st.title("Painel — atrasos de voos")
    st.caption("Resultados produzidos pelo pipeline principal (`python run_all.py`).")
    meta_p = OUT / "run_metadata.json"
    if not meta_p.exists():
        st.warning("Pasta outputs/ vazia ou ausente. Execute o pipeline principal antes de abrir o painel.")
        return
    meta = load_json(meta_p)
    st.subheader("Resumo da execucao")
    st.json(
        {
            "linhas_carregadas": meta.get("rows_loaded"),
            "linhas_modelagem": meta.get("rows_modeling"),
            "arquivo": meta.get("data_path"),
        }
    )
    cls = load_json(OUT / "supervised_classification" / "classification_metrics.json")
    if cls:
        st.subheader("Classificacao (atraso > 15 min)")
        st.write("Melhor modelo (ROC-AUC no holdout):", cls.get("best_by_roc_auc"))
        df_cls = pd.DataFrame(cls["per_model"]).T
        show_dataframe(df_cls)
    reg = load_json(OUT / "supervised_regression" / "regression_metrics.json")
    if reg:
        st.subheader("Regressao (atraso na partida, clip)")
        st.write("Melhor modelo (RMSE no holdout):", reg.get("best_by_rmse"))
        df_reg = pd.DataFrame(reg["per_model"]).T
        show_dataframe(df_reg)
    clus = load_json(OUT / "unsupervised" / "clustering_summary.json")
    if clus:
        st.subheader("Clusterizacao de aeroportos")
        st.json(clus)
    ano = load_json(OUT / "anomalies" / "anomaly_summary.json")
    if ano:
        st.subheader("Anomalias (aeroportos)")
        st.json(ano)
    semi = load_json(OUT / "semi_supervised" / "semi_supervised_metrics.json")
    if semi:
        st.subheader("Semi-supervisionado (SelfTraining + regressao logistica)")
        st.json(semi)
        psemi = OUT / "semi_supervised" / "semi_supervised_confusion_matrix.png"
        if psemi.exists():
            st.image(str(psemi))
    st.subheader("Artefatos")
    c1, c2 = st.columns(2)
    with c1:
        p = OUT / "eda_figures" / "hist_departure_delay.png"
        if p.exists():
            st.image(str(p))
    with c2:
        p2 = OUT / "unsupervised" / "pca_flights_2d.png"
        if p2.exists():
            st.image(str(p2))
    map_html = OUT / "maps" / "delay_mean_by_airport.html"
    if map_html.exists():
        st.subheader("Mapa — atraso medio por aeroporto")
        components.html(map_html.read_text(encoding="utf-8"), height=520, scrolling=True)
    routes_html = OUT / "maps" / "routes_top_od.html"
    if routes_html.exists():
        st.subheader("Mapa — principais rotas (origem-destino)")
        components.html(routes_html.read_text(encoding="utf-8"), height=520, scrolling=True)
    pst = OUT / "eda_figures" / "delay_rate_by_origin_state.png"
    if pst.exists():
        st.subheader("EDA — estado de origem")
        st.image(str(pst))
    miss = OUT / "eda_tables" / "missing_rate.csv"
    if miss.exists():
        st.subheader("Top variaveis com ausencia")
        show_dataframe(pd.read_csv(miss).head(25))


if __name__ == "__main__":
    main()
