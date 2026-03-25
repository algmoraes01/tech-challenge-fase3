import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from src.features import TopCategoryBucket
from src.supervised import make_preprocess, _build_xy


def train_semi_supervised_classification(
    df_work,
    out_dir: Path,
    random_state=42,
    test_size=0.2,
    unlabeled_fraction=0.55,
):
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
    rng = np.random.RandomState(random_state)
    y_semi = y_train.astype(int).copy()
    idx0 = np.flatnonzero(y_train == 0)
    idx1 = np.flatnonzero(y_train == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0 = int(len(idx0) * unlabeled_fraction)
    n1 = int(len(idx1) * unlabeled_fraction)
    y_semi[idx0[:n0]] = -1
    y_semi[idx1[:n1]] = -1
    n_labeled = int((y_semi != -1).sum())
    n_unlabeled = int((y_semi == -1).sum())
    base = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    st = SelfTrainingClassifier(base, threshold=0.75, max_iter=50, verbose=0)
    pipe = Pipeline([("prep", make_preprocess()), ("model", st)])
    pipe.fit(X_train, y_semi)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    summary = {
        "n_train_labeled": n_labeled,
        "n_train_unlabeled": n_unlabeled,
        "unlabeled_fraction_applied": unlabeled_fraction,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    (out_dir / "semi_supervised_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rep = classification_report(y_test, pred, digits=4)
    (out_dir / "semi_supervised_report.txt").write_text(rep, encoding="utf-8")
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Semi-supervisionado (SelfTraining + logistica)")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "semi_supervised_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return summary
