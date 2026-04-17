"""
src/preprocessing.py
====================
Módulo de preprocesamiento reutilizable para el proyecto de detección de phishing.

Encapsula las transformaciones definidas en el Sprint 2:
  - build_preprocessor(): construye el sklearn.Pipeline listo para fit/transform
  - load_data(): carga el dataset desde una ruta
  - split_data(): realiza el train/test split estratificado
  - apply_smote(): aplica SMOTE sobre el train set
  - save_pipeline(): serializa el pipeline con joblib
  - load_pipeline(): carga el pipeline desde disco

Uso desde un notebook:
    from src.preprocessing import build_preprocessor, load_data, split_data

Sprint 2 — Proyecto: Detección de Sitios Web Fraudulentos (Phishing)
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# =============================================================================
# Definición de grupos de variables
# =============================================================================

URL_FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service",
    "having_At_Symbol", "double_slash_redirecting",
    "Prefix_Suffix", "having_Sub_Domain"
]

SECURITY_FEATURES = [
    "SSLfinal_State", "Domain_registeration_length", "age_of_domain",
    "DNSRecord", "Favicon", "HTTPS_token"
]

CONTENT_FEATURES = [
    "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Redirect", "on_mouseover",
    "RightClick", "popUpWidnow", "Iframe"
]

POPULARITY_FEATURES = [
    "web_traffic", "Google_Index", "Page_Rank",
    "Links_pointing_to_page", "Statistical_report"
]

TARGET_COL = "Result"


# =============================================================================
# Carga de datos
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset desde la ruta indicada.

    Parameters
    ----------
    path : str
        Ruta al archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """
    df = pd.read_csv(path)
    print(f"[load_data] Dataset cargado desde '{path}'. Shape: {df.shape}")
    return df


# =============================================================================
# Limpieza básica
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza básica al dataset:
      - Elimina filas duplicadas.
      - Reinicia el índice.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo.

    Returns
    -------
    pd.DataFrame
        Dataset limpio.
    """
    df_clean = df.copy()
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    print(f"[clean_data] Duplicados eliminados: {n_removed}. Shape resultante: {df_clean.shape}")
    return df_clean


# =============================================================================
# Feature Engineering
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features derivadas a partir del dataset de phishing.

    Features añadidas:
      - url_risk_score: conteo de señales negativas en variables URL.
      - security_score: suma de variables de seguridad/dominio.
      - total_suspicious_count: conteo global de valores -1.
      - total_legitimate_count: conteo global de valores 1.
      - net_signal_ratio: diferencia entre señales positivas y negativas.
      - ssl_traffic_interaction: interacción SSL × tráfico.
      - content_risk_score: conteo de señales negativas en contenido.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset post-limpieza.

    Returns
    -------
    pd.DataFrame
        Dataset con features adicionales.
    """
    df_fe = df.copy()
    cols = df_fe.columns.tolist()

    # Filtrar variables disponibles en el dataset
    url_cols = [c for c in URL_FEATURES if c in cols]
    sec_cols = [c for c in SECURITY_FEATURES if c in cols]
    cnt_cols = [c for c in CONTENT_FEATURES if c in cols]

    feature_cols = [c for c in cols if c != TARGET_COL]

    # Features por grupo
    if url_cols:
        df_fe["url_risk_score"] = (df_fe[url_cols] == -1).sum(axis=1)

    if sec_cols:
        df_fe["security_score"] = df_fe[sec_cols].sum(axis=1)

    df_fe["total_suspicious_count"] = (df_fe[feature_cols] == -1).sum(axis=1)
    df_fe["total_legitimate_count"] = (df_fe[feature_cols] == 1).sum(axis=1)
    df_fe["net_signal_ratio"] = (
        df_fe["total_legitimate_count"] - df_fe["total_suspicious_count"]
    )

    # Interacción SSL × tráfico
    if "SSLfinal_State" in cols and "web_traffic" in cols:
        df_fe["ssl_traffic_interaction"] = (
            df_fe["SSLfinal_State"] * df_fe["web_traffic"]
        )

    if cnt_cols:
        df_fe["content_risk_score"] = (df_fe[cnt_cols] == -1).sum(axis=1)

    new_feat = [c for c in df_fe.columns if c not in cols]
    print(f"[add_features] Features añadidas ({len(new_feat)}): {new_feat}")
    return df_fe


# =============================================================================
# Train/Test Split
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Realiza el train/test split estratificado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo con features y target.
    target_col : str
        Nombre de la columna objetivo.
    test_size : float
        Proporción del test set.
    random_state : int
        Semilla de aleatoriedad.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"[split_data] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Balance train: {dict(y_train.value_counts().sort_index())}")
    print(f"  Balance test:  {dict(y_test.value_counts().sort_index())}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# Balanceo de clases (SMOTE)
# =============================================================================

def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Aplica SMOTE sobre el conjunto de entrenamiento.

    ⚠️ Siempre llamar después de split_data(). Nunca sobre el test set.

    Parameters
    ----------
    X_train : array-like
        Features de entrenamiento.
    y_train : array-like
        Target de entrenamiento.
    random_state : int
        Semilla de aleatoriedad.

    Returns
    -------
    tuple : (X_train_bal, y_train_bal)
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    print(f"[apply_smote] Antes:  {dict(pd.Series(y_train).value_counts().sort_index())}")
    print(f"[apply_smote] Después:{dict(pd.Series(y_bal).value_counts().sort_index())}")

    return X_bal, y_bal


# =============================================================================
# Pipeline de preprocesamiento
# =============================================================================

def build_preprocessor(numeric_cols: list) -> ColumnTransformer:
    """
    Construye el ColumnTransformer con el pipeline de preprocesamiento.

    Pasos por variable numérica:
      1. SimpleImputer(strategy='median') — robustez ante nulos en producción.
      2. StandardScaler() — normalización a media=0, std=1.

    Parameters
    ----------
    numeric_cols : list
        Lista de columnas numéricas a transformar.

    Returns
    -------
    ColumnTransformer
        Pipeline de preprocesamiento listo para fit/transform.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols)
        ],
        remainder="drop"
    )

    print(f"[build_preprocessor] Pipeline construido para {len(numeric_cols)} columnas numéricas.")
    return preprocessor


# =============================================================================
# Persistencia del pipeline
# =============================================================================

def save_pipeline(pipeline, path: str) -> None:
    """
    Serializa el pipeline con joblib.

    Parameters
    ----------
    pipeline : sklearn estimator
        Pipeline entrenado.
    path : str
        Ruta de destino (.pkl).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[save_pipeline] Pipeline guardado en '{path}'.")


def load_pipeline(path: str):
    """
    Carga el pipeline serializado desde disco.

    Parameters
    ----------
    path : str
        Ruta al archivo .pkl.

    Returns
    -------
    sklearn estimator
        Pipeline cargado.
    """
    pipeline = joblib.load(path)
    print(f"[load_pipeline] Pipeline cargado desde '{path}'.")
    return pipeline


# =============================================================================
# Función de preprocesamiento completo (end-to-end)
# =============================================================================

def full_preprocessing_pipeline(
    raw_path: str,
    output_dir: str = "../data/processed",
    model_dir: str = "../models",
    apply_balance: bool = True,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Ejecuta el pipeline completo de preprocesamiento:
    carga → limpieza → features → split → (SMOTE) → fit/transform → guardar.

    Parameters
    ----------
    raw_path : str
        Ruta al dataset crudo.
    output_dir : str
        Directorio donde guardar los CSV procesados.
    model_dir : str
        Directorio donde guardar el pipeline serializado.
    apply_balance : bool
        Si True, aplica SMOTE sobre el train set.
    test_size : float
        Proporción del test set.
    random_state : int
        Semilla de aleatoriedad.

    Returns
    -------
    dict con claves:
        X_train, X_test, y_train, y_test,
        X_train_t, X_test_t,  (transformados)
        preprocessor
    """
    # 1. Carga
    df = load_data(raw_path)

    # 2. Limpieza
    df = clean_data(df)

    # 3. Feature engineering
    df = add_features(df)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )

    # 5. Balanceo (opcional)
    if apply_balance:
        X_train_bal, y_train_bal = apply_smote(X_train, y_train, random_state)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # 6. Construir y entrenar pipeline
    numeric_cols = X_train.columns.tolist()
    preprocessor = build_preprocessor(numeric_cols)
    preprocessor.fit(X_train_bal)

    X_train_t = preprocessor.transform(X_train_bal)
    X_test_t = preprocessor.transform(X_test)

    # 7. Guardar archivos
    os.makedirs(output_dir, exist_ok=True)

    train_out = pd.DataFrame(X_train_t, columns=numeric_cols)
    train_out[TARGET_COL] = y_train_bal if hasattr(y_train_bal, "values") else list(y_train_bal)
    train_out.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)

    test_out = pd.DataFrame(X_test_t, columns=numeric_cols)
    test_out[TARGET_COL] = y_test.values
    test_out.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)

    save_pipeline(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))

    print("\n[full_preprocessing_pipeline] ✓ Preprocesamiento completo finalizado.")
    print(f"  Train transformado: {X_train_t.shape}")
    print(f"  Test transformado:  {X_test_t.shape}")

    return {
        "X_train": X_train_bal,
        "X_test": X_test,
        "y_train": y_train_bal,
        "y_test": y_test,
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "preprocessor": preprocessor
    }
