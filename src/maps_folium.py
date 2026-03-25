import math
from pathlib import Path

import folium
import pandas as pd
from branca.colormap import LinearColormap

from src.features import airport_delay_aggregates


def build_route_map(df_work, coords_df: pd.DataFrame, out_html: Path, top_n=120):
    if coords_df is None or coords_df.empty:
        return None
    c = coords_df.rename(columns={"iata": "code"}).drop_duplicates(subset="code")
    routes = (
        df_work.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], as_index=False)
        .agg(n_flights=("FLIGHT_NUMBER", "count"), mean_delay=("DEPARTURE_DELAY", "mean"))
        .nlargest(top_n, "n_flights")
    )
    o = c.rename(columns={"code": "ORIGIN_AIRPORT", "lat": "o_lat", "lon": "o_lon"})
    d = c.rename(columns={"code": "DESTINATION_AIRPORT", "lat": "d_lat", "lon": "d_lon"})
    r = routes.merge(o, on="ORIGIN_AIRPORT", how="inner").merge(d, on="DESTINATION_AIRPORT", how="inner")
    if r.empty:
        return None
    center_lat = float(r[["o_lat", "d_lat"]].mean(axis=1).median())
    center_lon = float(r[["o_lon", "d_lon"]].mean(axis=1).median())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="cartodbpositron")
    dmin = float(r["mean_delay"].quantile(0.1))
    dmax = float(r["mean_delay"].quantile(0.9))
    span = max(dmax - dmin, 1.0)
    cmap = LinearColormap(["#2166ac", "#f7f7f7", "#b2182b"], vmin=dmin, vmax=dmin + span)
    for _, row in r.iterrows():
        col = cmap(float(row["mean_delay"]))
        w = 1 + 2.0 * math.log1p(float(row["n_flights"])) / math.log1p(float(r["n_flights"].max()))
        folium.PolyLine(
            locations=[
                [float(row["o_lat"]), float(row["o_lon"])],
                [float(row["d_lat"]), float(row["d_lon"])],
            ],
            color=col,
            weight=max(1, min(8, w)),
            opacity=0.55,
            popup=f"{row['ORIGIN_AIRPORT']} → {row['DESTINATION_AIRPORT']}: n={int(row['n_flights'])}, atraso medio {row['mean_delay']:.1f} min",
        ).add_to(fmap)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


def build_delay_map(df_work, airports_geo_path: Path, out_html: Path):
    g = airport_delay_aggregates(df_work)
    geo = pd.read_csv(airports_geo_path)
    geo = geo.rename(columns={"iata": "airport"})
    mrg = g.merge(geo, on="airport", how="inner")
    if mrg.empty:
        return None
    center_lat = float(mrg["lat"].median())
    center_lon = float(mrg["lon"].median())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="cartodbpositron")
    vmax = float(mrg["mean_dep_delay"].quantile(0.95))
    vmax = max(vmax, 1.0)
    cmap = LinearColormap(["#1a9850", "#fee08b", "#d73027"], vmin=0.0, vmax=vmax)
    for _, r in mrg.iterrows():
        raw = float(r["mean_dep_delay"])
        val_c = max(0.0, min(raw, vmax))
        color = cmap(val_c)
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=4 + min(18, float(r["n_flights"]) / 5000),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=f"{r['airport']}: atraso medio {raw:.1f} min, voos {int(r['n_flights'])}",
        ).add_to(fmap)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html
