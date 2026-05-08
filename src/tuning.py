"""
src/tuning.py
Sprint 4 – Hyperparameter Tuning Utilities
Equipo 5 – Phishing Detection
"""

import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score


# ─────────────────────────────────────────
# CV estratificado global (reutilizable)
# ─────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    "f1":        "f1",
    "roc_auc":   "roc_auc",
    "recall":    "recall",
    "precision": "precision",
}


# ─────────────────────────────────────────
# tune_model: GridSearchCV wrapper
# ─────────────────────────────────────────
def tune_model(pipe, param_grid, X, y,
               cv=CV, scoring="f1", name="model"):
    """
    Tunea hiperparámetros con GridSearchCV y persiste el mejor estimador.

    Parameters
    ----------
    pipe       : sklearn Pipeline o estimador
    param_grid : dict  – espacio de búsqueda
    X, y       : datos de entrenamiento
    cv         : cross-validator
    scoring    : métrica principal
    name       : identificador del modelo (usado en el nombre del .pkl)

    Returns
    -------
    dict con model, best_params, best_score, time_seconds, path
    """
    start = time.time()
    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=-1, verbose=0,
        refit=True
    )
    grid.fit(X, y)
    elapsed = round(time.time() - start, 1)

    path = f"../models/tuned_{name}.pkl"
    joblib.dump(grid.best_estimator_, path)

    return {
        "model":        name,
        "method":       "GridSearchCV",
        "best_params":  grid.best_params_,
        "best_score":   round(grid.best_score_, 4),
        "time_seconds": elapsed,
        "path":         path,
    }


# ─────────────────────────────────────────
# tune_model_random: RandomizedSearchCV wrapper
# ─────────────────────────────────────────
def tune_model_random(pipe, param_dist, X, y,
                      n_iter=100, cv=CV, scoring="f1", name="model"):
    """
    Tunea con RandomizedSearchCV (recomendado para espacios grandes).
    """
    start = time.time()
    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=n_iter,
        cv=cv, scoring=scoring,
        n_jobs=-1, verbose=0,
        random_state=42, refit=True
    )
    search.fit(X, y)
    elapsed = round(time.time() - start, 1)

    path = f"../models/tuned_{name}.pkl"
    joblib.dump(search.best_estimator_, path)

    return {
        "model":        name,
        "method":       "RandomizedSearchCV",
        "best_params":  search.best_params_,
        "best_score":   round(search.best_score_, 4),
        "time_seconds": elapsed,
        "path":         path,
    }


# ─────────────────────────────────────────
# log_experiment: registra en experiments_log.csv
# ─────────────────────────────────────────
def log_experiment(result: dict, log_path="../models/experiments_log.csv"):
    """
    Agrega una fila al CSV de experimentos del proyecto.
    Crea el archivo si no existe.
    """
    row = {
        "sprint":       4,
        "model":        result.get("model"),
        "method":       result.get("method"),
        "best_score":   result.get("best_score"),
        "time_seconds": result.get("time_seconds"),
        "path":         result.get("path"),
        "params":       str(result.get("best_params")),
    }
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log_path, index=False)
    print(f"[log] Experimento '{result['model']}' registrado en {log_path}")


# ─────────────────────────────────────────
# compare_metrics: baseline vs tuned
# ─────────────────────────────────────────
def compare_metrics(baseline_model, tuned_model, X_test, y_test):
    """
    Genera tabla comparativa de métricas entre baseline y modelo tuneado.
    """
    results = {}
    for label, model in [("Baseline (Sprint 3)", baseline_model),
                         ("Tuned (Sprint 4)", tuned_model)]:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[label] = {
            "F1":        round(f1_score(y_test, y_pred), 4),
            "AUC-ROC":   round(roc_auc_score(y_test, y_proba), 4),
            "Recall":    round(recall_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
        }

    df = pd.DataFrame(results).T
    df["Mejora (%)"] = (
        (df.loc["Tuned (Sprint 4)"] - df.loc["Baseline (Sprint 3)"])
        / df.loc["Baseline (Sprint 3)"] * 100
    ).round(1)
    return df


# ─────────────────────────────────────────
# bootstrap_metric: intervalos de confianza
# ─────────────────────────────────────────
def bootstrap_metric(y_true, y_score, metric_fn=roc_auc_score, n=1000, ci=95):
    """
    Calcula intervalo de confianza de una métrica via bootstrap.

    Returns
    -------
    (lower, upper)
    """
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    scores  = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    alpha = (100 - ci) / 2
    return tuple(np.percentile(scores, [alpha, 100 - alpha]).round(4))


# ─────────────────────────────────────────
# load_model: carga un modelo persistido
# ─────────────────────────────────────────
def load_model(path: str):
    """Carga y devuelve un modelo guardado con joblib."""
    model = joblib.load(path)
    print(f"[load] Modelo cargado desde: {path}")
    return model