from pathlib import Path

import numpy as np
import pandas as pd


def season_from_month(month):
    m = int(month)
    if m in (12, 1, 2):
        return 1
    if m in (3, 4, 5):
        return 2
    if m in (6, 7, 8):
        return 3
    return 4


def load_flights(path, nrows=None):
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def merge_airport_master(df, airports_csv_path):
    path = Path(airports_csv_path)
    if not path.exists():
        out = df.copy()
        out["ORIGIN_STATE"] = "UNK"
        out["DESTINATION_STATE"] = "UNK"
        return out
    ap = pd.read_csv(path, low_memory=False)
    if "IATA_CODE" not in ap.columns or "STATE" not in ap.columns:
        out = df.copy()
        out["ORIGIN_STATE"] = "UNK"
        out["DESTINATION_STATE"] = "UNK"
        return out
    m = ap[["IATA_CODE", "STATE"]].drop_duplicates(subset="IATA_CODE")
    mo = m.rename(columns={"IATA_CODE": "ORIGIN_AIRPORT", "STATE": "ORIGIN_STATE"})
    out = df.merge(mo, on="ORIGIN_AIRPORT", how="left")
    md = m.rename(columns={"IATA_CODE": "DESTINATION_AIRPORT", "STATE": "DESTINATION_STATE"})
    out = out.merge(md, on="DESTINATION_AIRPORT", how="left")
    out["ORIGIN_STATE"] = out["ORIGIN_STATE"].fillna("UNK").astype(str)
    out["DESTINATION_STATE"] = out["DESTINATION_STATE"].fillna("UNK").astype(str)
    return out


def add_us_holiday_features(df):
    import holidays

    out = df.copy()
    ys = pd.to_numeric(out["YEAR"], errors="coerce").dropna()
    if ys.empty:
        out["is_us_holiday"] = 0
        out["is_day_before_us_holiday"] = 0
        return out
    y0, y1 = int(ys.min()), int(ys.max())
    cal = holidays.US(years=range(y0, y1 + 1))
    dt = pd.to_datetime(
        {
            "year": pd.to_numeric(out["YEAR"], errors="coerce"),
            "month": pd.to_numeric(out["MONTH"], errors="coerce"),
            "day": pd.to_numeric(out["DAY"], errors="coerce"),
        },
        errors="coerce",
    )
    d_only = dt.dt.date
    uniq_d = pd.unique(d_only.dropna())
    hol_today = {d: int(d in cal) for d in uniq_d}
    out["is_us_holiday"] = d_only.map(lambda x: hol_today.get(x, 0) if pd.notna(x) else 0).astype(int)
    nxt = (dt + pd.Timedelta(days=1)).dt.date
    uniq_n = pd.unique(nxt.dropna())
    hol_next = {d: int(d in cal) for d in uniq_n}
    mask = pd.notna(dt)
    out["is_day_before_us_holiday"] = 0
    s_n = nxt.loc[mask]
    out.loc[mask, "is_day_before_us_holiday"] = s_n.map(lambda x: hol_next.get(x, 0))
    out["is_day_before_us_holiday"] = out["is_day_before_us_holiday"].astype(int)
    return out


def load_airport_coordinates(airports_csv_path):
    path = Path(airports_csv_path)
    if not path.exists():
        return None
    ap = pd.read_csv(path, low_memory=False)
    need = {"IATA_CODE", "LATITUDE", "LONGITUDE"}
    if not need.issubset(ap.columns):
        return None
    g = ap[list(need)].drop_duplicates(subset="IATA_CODE").rename(
        columns={"IATA_CODE": "iata", "LATITUDE": "lat", "LONGITUDE": "lon"}
    )
    g = g.dropna(subset=["lat", "lon"])
    return g


def filter_for_delay_model(df):
    d = df.loc[df["CANCELLED"] == 0].copy()
    if "DIVERTED" in d.columns:
        d = d.loc[d["DIVERTED"] == 0]
    d = d.dropna(subset=["DEPARTURE_DELAY"])
    return d


def add_derived_columns(df):
    out = df.copy()
    sd = pd.to_numeric(out["SCHEDULED_DEPARTURE"], errors="coerce").fillna(0).astype(int)
    out["dep_hour"] = (sd // 100).clip(0, 23)
    out["dep_minute"] = (sd % 100).clip(0, 59)
    out["season"] = out["MONTH"].apply(season_from_month)
    out["is_weekend"] = (out["DAY_OF_WEEK"] >= 6).astype(int)
    out["is_rush_morning"] = ((out["dep_hour"] >= 6) & (out["dep_hour"] <= 9)).astype(int)
    out["is_rush_evening"] = ((out["dep_hour"] >= 16) & (out["dep_hour"] <= 20)).astype(int)
    out["sin_dow"] = np.sin(2 * np.pi * (out["DAY_OF_WEEK"] - 1) / 7)
    out["cos_dow"] = np.cos(2 * np.pi * (out["DAY_OF_WEEK"] - 1) / 7)
    out["sin_month"] = np.sin(2 * np.pi * (out["MONTH"] - 1) / 12)
    out["cos_month"] = np.cos(2 * np.pi * (out["MONTH"] - 1) / 12)
    out["sin_hour"] = np.sin(2 * np.pi * out["dep_hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["dep_hour"] / 24)
    return out


class TopCategoryBucket:
    def __init__(self, column, top_n=45):
        self.column = column
        self.top_n = top_n
        self._keep = None

    def fit(self, series):
        vc = series.value_counts()
        self._keep = set(vc.nlargest(self.top_n).index)

    def transform(self, series):
        return series.where(series.isin(self._keep), "__OTHER__")


def prepare_classification_targets(df, threshold_minutes=15):
    y = (df["DEPARTURE_DELAY"] > threshold_minutes).astype(int)
    return y


def airport_delay_aggregates(df):
    def _pct_gt15(s):
        return float((s > 15).mean())

    g = (
        df.groupby("ORIGIN_AIRPORT", as_index=False)
        .agg(
            n_flights=("FLIGHT_NUMBER", "count"),
            mean_dep_delay=("DEPARTURE_DELAY", "mean"),
            median_dep_delay=("DEPARTURE_DELAY", "median"),
            pct_gt15=("DEPARTURE_DELAY", _pct_gt15),
        )
    )
    g = g.rename(columns={"ORIGIN_AIRPORT": "airport"})
    return g
