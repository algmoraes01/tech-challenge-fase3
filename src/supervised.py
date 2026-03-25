import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.features import TopCategoryBucket


def _build_xy(df, origin_buck: TopCategoryBucket, dest_buck: TopCategoryBucket):
    d = df.copy()
    for col in ("ORIGIN_STATE", "DESTINATION_STATE"):
        if col not in d.columns:
            d[col] = "UNK"
    for col in ("is_us_holiday", "is_day_before_us_holiday"):
        if col not in d.columns:
            d[col] = 0
    d["ORIGIN_AIRPORT"] = origin_buck.transform(d["ORIGIN_AIRPORT"])
    d["DESTINATION_AIRPORT"] = dest_buck.transform(d["DESTINATION_AIRPORT"])
    num_cols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "SCHEDULED_TIME",
        "DISTANCE",
        "dep_hour",
        "dep_minute",
        "season",
        "is_weekend",
        "is_rush_morning",
        "is_rush_evening",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "sin_hour",
        "cos_hour",
        "is_us_holiday",
        "is_day_before_us_holiday",
    ]
    cat_cols = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ORIGIN_STATE",
        "DESTINATION_STATE",
    ]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=num_cols)
    X = d[num_cols + cat_cols]
    return X, d.index


def make_preprocess():
    num_cols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "SCHEDULED_TIME",
        "DISTANCE",
        "dep_hour",
        "dep_minute",
        "season",
        "is_weekend",
        "is_rush_morning",
        "is_rush_evening",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "sin_hour",
        "cos_hour",
        "is_us_holiday",
        "is_day_before_us_holiday",
    ]
    cat_cols = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ORIGIN_STATE",
        "DESTINATION_STATE",
    ]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=65, sparse_output=False), cat_cols),
        ]
    )


def train_classification(df_work, out_dir: Path, random_state=42, test_size=0.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    y = (df_work["DEPARTURE_DELAY"] > 15).astype(int).loc[df_work.index]
    origin_buck = TopCategoryBucket("ORIGIN_AIRPORT", top_n=45)
    dest_buck = TopCategoryBucket("DESTINATION_AIRPORT", top_n=45)
    origin_buck.fit(df_work["ORIGIN_AIRPORT"])
    dest_buck.fit(df_work["DESTINATION_AIRPORT"])
    X, idx = _build_xy(df_work, origin_buck, dest_buck)
    y = y.loc[idx].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = max(neg / max(pos, 1.0), 1.0)
    sample_w = compute_sample_weight("balanced", y_train)

    def build_estimators():
        return {
            "logistic_regression": LogisticRegression(
                max_iter=400,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=50,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=spw,
        ),
        }

    results = {}
    for name, est in build_estimators().items():
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        cv_pipe = Pipeline([("prep", make_preprocess()), ("model", est)])
        cv_auc = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        est = build_estimators()[name]
        pipe = Pipeline([("prep", make_preprocess()), ("model", est)])
        if name == "gradient_boosting":
            pipe.fit(X_train, y_train, model__sample_weight=sample_w)
        else:
            pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        }
        results[name] = row
        results[name]["cv_roc_auc_mean"] = float(np.mean(cv_auc))
        results[name]["cv_roc_auc_std"] = float(np.std(cv_auc))
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusao — {name}")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)
        rep = classification_report(y_test, pred, digits=4)
        (out_dir / f"classification_report_{name}.txt").write_text(rep, encoding="utf-8")
        joblib.dump(pipe, out_dir / f"pipeline_cls_{name}.joblib")
    best = max(results, key=lambda k: results[k]["roc_auc"])
    summary = {"per_model": results, "best_by_roc_auc": best}
    (out_dir / "classification_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, origin_buck, dest_buck


def train_regression(df_work, out_dir: Path, random_state=42, test_size=0.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = df_work.copy()
    d["dep_delay_clip"] = d["DEPARTURE_DELAY"].clip(-50, 180)
    y_all = d["dep_delay_clip"]
    origin_buck = TopCategoryBucket("ORIGIN_AIRPORT", top_n=45)
    dest_buck = TopCategoryBucket("DESTINATION_AIRPORT", top_n=45)
    origin_buck.fit(d["ORIGIN_AIRPORT"])
    dest_buck.fit(d["DESTINATION_AIRPORT"])
    X, idx = _build_xy(d, origin_buck, dest_buck)
    y = y_all.loc[idx].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def build_regressors():
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=50,
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
            "xgboost": XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                tree_method="hist",
            ),
        }

    results = {}
    for name, est in build_regressors().items():
        pipe = Pipeline([("prep", make_preprocess()), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        row = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, pred)),
        }
        results[name] = row
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, pred, alpha=0.08, s=8, c="steelblue")
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Real (clip)")
        ax.set_ylabel("Predito")
        ax.set_title(f"Regressao — {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"reg_scatter_{name}.png", dpi=150)
        plt.close(fig)
        joblib.dump(pipe, out_dir / f"pipeline_reg_{name}.joblib")
    best = min(results, key=lambda k: results[k]["rmse"])
    summary = {"per_model": results, "best_by_rmse": best}
    (out_dir / "regression_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
