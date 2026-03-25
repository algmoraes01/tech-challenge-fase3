import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.anomalies import run_isolation_forest_airports
from src.eda import describe_and_save, plot_eda
from src.features import (
    add_derived_columns,
    add_us_holiday_features,
    filter_for_delay_model,
    load_airport_coordinates,
    load_flights,
    merge_airport_master,
)
from src.maps_folium import build_delay_map, build_route_map
from src.semisupervised import train_semi_supervised_classification
from src.supervised import train_classification, train_regression
from src.unsupervised import run_airport_clustering, run_pca_on_flights


def impute_missing_for_model(df):
    d = df.copy()
    if "SCHEDULED_TIME" in d.columns:
        d["SCHEDULED_TIME"] = pd.to_numeric(d["SCHEDULED_TIME"], errors="coerce")
        d["SCHEDULED_TIME"] = d["SCHEDULED_TIME"].fillna(d["SCHEDULED_TIME"].median())
    if "DISTANCE" in d.columns:
        d["DISTANCE"] = pd.to_numeric(d["DISTANCE"], errors="coerce")
        d["DISTANCE"] = d["DISTANCE"].fillna(d["DISTANCE"].median())
    if "TAIL_NUMBER" in d.columns:
        d["TAIL_NUMBER"] = d["TAIL_NUMBER"].fillna("UNK")
    return d


def main():
    data_path = os.environ.get("FLIGHTS_CSV", str(ROOT / "data" / "flights.csv"))
    nrows = os.environ.get("FLIGHTS_NROWS")
    nrows = int(nrows) if nrows else None
    out_root = ROOT / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    if not Path(data_path).exists():
        raise SystemExit(
            f"Arquivo não encontrado: {data_path}. Coloque flights.csv em data/ ou defina a variável FLIGHTS_CSV."
        )
    ap_path = ROOT / "data" / "airports.csv"
    df = load_flights(data_path, nrows=nrows)
    df = merge_airport_master(df, ap_path)
    df = add_us_holiday_features(df)
    describe_and_save(df, out_root / "eda_tables")
    plot_eda(df, out_root / "eda_figures")
    df_model = filter_for_delay_model(df)
    df_model = impute_missing_for_model(df_model)
    df_model = add_derived_columns(df_model)
    run_pca_on_flights(df_model, out_root / "unsupervised")
    clus_sum, _ = run_airport_clustering(df_model, out_root / "unsupervised")
    ano_sum = run_isolation_forest_airports(df_model, out_root / "anomalies")
    cls_sum, _, _ = train_classification(df_model, out_root / "supervised_classification")
    reg_sum = train_regression(df_model, out_root / "supervised_regression")
    semi_sum = train_semi_supervised_classification(df_model, out_root / "semi_supervised")
    geo_path = ROOT / "data" / "airports_geo.csv"
    map_path = out_root / "maps" / "delay_mean_by_airport.html"
    built = build_delay_map(df_model, geo_path, map_path)
    coords = load_airport_coordinates(ap_path)
    route_built = None
    if coords is not None:
        route_built = build_route_map(df_model, coords, out_root / "maps" / "routes_top_od.html")
    meta = {
        "rows_loaded": int(len(df)),
        "rows_modeling": int(len(df_model)),
        "data_path": str(Path(data_path).resolve()),
        "nrows_limit": nrows,
        "clustering": clus_sum,
        "anomalies": ano_sum,
        "classification": cls_sum,
        "regression": reg_sum,
        "semi_supervised": semi_sum,
        "map_html": str(built) if built else None,
        "routes_map_html": str(route_built) if route_built else None,
    }
    (out_root / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
