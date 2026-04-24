"""
src/models.py
=============
Módulo reutilizable de entrenamiento, evaluación y registro de experimentos.
Sprint 3 — Proyecto: Detección de Sitios Web Fraudulentos (Phishing)

Funciones:
    get_baseline_models()    : devuelve dict con los 6 clasificadores baseline
    build_full_pipeline()    : envuelve preproc + clf en un sklearn Pipeline
    train_baseline()         : entrena y persiste un modelo
    evaluate_model()         : cross-validation estratificada con múltiples métricas
    evaluate_on_test()       : evaluación final en test set (usar solo una vez)
    log_experiment()         : agrega una fila al CSV de experimentos
    compare_models()         : entrena y evalúa todos los baselines, retorna DataFrame
    load_model()             : carga un modelo serializado con joblib

Uso desde un notebook:
    import sys; sys.path.insert(0, '..')
    from src.models import get_baseline_models, compare_models, log_experiment
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Clasificadores baseline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# =============================================================================
# Configuración de métricas
# =============================================================================

CV_SCORING = {
    "accuracy":  "accuracy",
    "f1":        "f1",
    "precision": "precision",
    "recall":    "recall",
    "roc_auc":   "roc_auc",
}

CV_FOLDS = 5
RANDOM_STATE = 42


# =============================================================================
# Definición de modelos baseline
# =============================================================================

def get_baseline_models() -> dict:
    """
    Retorna un diccionario con los clasificadores baseline en parámetros por defecto.

    Returns
    -------
    dict
        {nombre: estimador_sklearn}
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
    }


# =============================================================================
# Pipeline completo: preprocesamiento + clasificador
# =============================================================================

def build_full_pipeline(preprocessor, classifier) -> Pipeline:
    """
    Construye un Pipeline sklearn que encadena preprocesamiento y clasificador.

    ⚠️ Regla de oro (Sprint 3): siempre pasar datos CRUDOS al Pipeline.
    cross_validate aplica fit(preproc) + fit(clf) en train y transform(preproc)
    + predict(clf) en val — evitando data leakage.

    Parameters
    ----------
    preprocessor : sklearn transformer
        Pipeline/ColumnTransformer del Sprint 2.
    classifier : sklearn estimator
        Clasificador baseline.

    Returns
    -------
    Pipeline
        Pipeline completo listo para fit/predict/cross_validate.
    """
    return Pipeline([
        ("preproc", preprocessor),
        ("clf",     classifier),
    ])


# =============================================================================
# Entrenamiento y persistencia
# =============================================================================

def train_baseline(pipe: Pipeline, X, y, name: str,
                   model_dir: str = "../models") -> Pipeline:
    """
    Entrena el Pipeline completo y lo persiste con joblib.

    Parameters
    ----------
    pipe : Pipeline
        Pipeline preproc + clf.
    X : array-like
        Features (datos crudos sin transformar).
    y : array-like
        Target.
    name : str
        Identificador corto del modelo (ej. 'lr', 'rf').
    model_dir : str
        Directorio donde guardar el .pkl.

    Returns
    -------
    Pipeline
        Pipeline entrenado.
    """
    pipe.fit(X, y)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"baseline_{name}.pkl")
    joblib.dump(pipe, path)
    print(f"[train_baseline] '{name}' entrenado y guardado en '{path}'.")
    return pipe


# =============================================================================
# Evaluación con cross-validation
# =============================================================================

def evaluate_model(pipe: Pipeline, X, y,
                   cv_folds: int = CV_FOLDS) -> dict:
    """
    Evalúa un Pipeline con StratifiedKFold cross-validation.

    Parameters
    ----------
    pipe : Pipeline
        Pipeline completo (preproc + clf).
    X : array-like
        Features (datos crudos).
    y : array-like
        Target.
    cv_folds : int
        Número de folds para cross-validation.

    Returns
    -------
    dict
        Resultados de cross_validate (test y train scores por métrica).
    """
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
    )
    results = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=CV_SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    return results


# =============================================================================
# Evaluación final en test set (usar solo una vez al finalizar el Sprint 3)
# =============================================================================

def evaluate_on_test(pipe: Pipeline, X_test, y_test) -> dict:
    """
    Evaluación definitiva en el test set.

    ⚠️ Llamar solo una vez al concluir la comparación de baselines.
    No usar para tomar decisiones de selección de modelo.

    Parameters
    ----------
    pipe : Pipeline
        Pipeline entrenado.
    X_test : array-like
        Features del test set (datos crudos).
    y_test : array-like
        Target del test set.

    Returns
    -------
    dict
        {'report': str, 'confusion_matrix': np.ndarray, 'roc_auc': float}
    """
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Phishing (-1)", "Legítimo (1)"])
    cm     = confusion_matrix(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(report)
    print(f"AUC-ROC (test): {auc:.4f}")
    return {"report": report, "confusion_matrix": cm, "roc_auc": auc}


# =============================================================================
# Registro de experimentos
# =============================================================================

def log_experiment(name: str, params: dict, metrics: dict,
                   notes: str = "",
                   path: str = "../models/experiments_log.csv") -> None:
    """
    Agrega una fila al CSV de registro de experimentos.

    Parameters
    ----------
    name : str
        Nombre del modelo.
    params : dict
        Parámetros del modelo (ej. {'n_estimators': 100}).
    metrics : dict
        Métricas obtenidas (F1, AUC, etc.).
    notes : str
        Observaciones libres sobre el experimento.
    path : str
        Ruta al CSV de registro.
    """
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sprint":    3,
        "model":     name,
        **{f"param_{k}": v for k, v in params.items()},
        **metrics,
        "notes":     notes,
    }
    df_row = pd.DataFrame([row])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=header, index=False)
    print(f"[log_experiment] '{name}' registrado en '{path}'.")


# =============================================================================
# Comparación de todos los baselines (flujo completo)
# =============================================================================

def compare_models(preprocessor, X_train, y_train,
                   model_dir: str = "../models",
                   log_path: str = "../models/experiments_log.csv") -> pd.DataFrame:
    """
    Entrena y evalúa todos los baselines. Retorna tabla comparativa.

    Flujo por modelo:
        build_full_pipeline → train_baseline → evaluate_model → log_experiment

    Parameters
    ----------
    preprocessor : sklearn transformer
        Pipeline de preprocesamiento del Sprint 2.
    X_train : DataFrame
        Features de entrenamiento (datos crudos, con SMOTE aplicado si corresponde).
    y_train : Series
        Target de entrenamiento.
    model_dir : str
        Directorio para persistir modelos.
    log_path : str
        Ruta al CSV de registro.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa con métricas de cross-validation, ordenada por F1 desc.
    """
    classifiers = get_baseline_models()
    records = []

    for full_name, clf in classifiers.items():
        short_name = full_name.lower().replace(" ", "_")
        print(f"\n{'='*60}")
        print(f"  Modelo: {full_name}")
        print(f"{'='*60}")

        # Pipeline completo
        pipe = build_full_pipeline(preprocessor, clf)

        # Entrenamiento + persistencia
        train_baseline(pipe, X_train, y_train, short_name, model_dir)

        # Cross-validation
        cv_results = evaluate_model(pipe, X_train, y_train)

        # Métricas de test CV (media ± std)
        metrics_cv = {}
        for metric in CV_SCORING:
            mean = cv_results[f"test_{metric}"].mean()
            std  = cv_results[f"test_{metric}"].std()
            metrics_cv[metric] = mean
            metrics_cv[f"{metric}_std"] = std

        # Detección de overfitting: train_f1 >> test_f1
        train_f1 = cv_results["train_f1"].mean()
        test_f1  = cv_results["test_f1"].mean()
        overfit_gap = train_f1 - test_f1
        overfit_flag = overfit_gap > 0.10

        row = {
            "Modelo":       full_name,
            "Accuracy":     round(metrics_cv["accuracy"], 4),
            "F1":           round(metrics_cv["f1"], 4),
            "F1_std":       round(metrics_cv["f1_std"], 4),
            "AUC-ROC":      round(metrics_cv["roc_auc"], 4),
            "Recall":       round(metrics_cv["recall"], 4),
            "Precision":    round(metrics_cv["precision"], 4),
            "Train_F1":     round(train_f1, 4),
            "Overfit_Gap":  round(overfit_gap, 4),
            "Overfitting?": "⚠️ Sí" if overfit_flag else "OK",
        }
        records.append(row)

        # Registro
        log_experiment(
            name=full_name,
            params=clf.get_params(),
            metrics=metrics_cv,
            notes=f"Baseline Sprint 3. Overfit gap: {overfit_gap:.3f}",
            path=log_path,
        )

    df = pd.DataFrame(records).sort_values("F1", ascending=False).reset_index(drop=True)
    df.index += 1
    print("\n\n✅ Comparación de baselines completada.")
    return df


# =============================================================================
# Carga de modelo persistido
# =============================================================================

def load_model(name: str, model_dir: str = "../models") -> Pipeline:
    """
    Carga un modelo baseline serializado.

    Parameters
    ----------
    name : str
        Identificador corto del modelo (ej. 'random_forest').
    model_dir : str
        Directorio donde está el .pkl.

    Returns
    -------
    Pipeline
        Pipeline entrenado listo para predict.
    """
    path = os.path.join(model_dir, f"baseline_{name}.pkl")
    pipe = joblib.load(path)
    print(f"[load_model] '{name}' cargado desde '{path}'.")
    return pipe
