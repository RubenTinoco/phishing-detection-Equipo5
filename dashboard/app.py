import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


st.set_page_config(
    page_title="Phishing Detection Platform",
    page_icon="shield",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.pkl"
SUMMARY_PATH = ROOT / "reports" / "business_value_summary.csv"
SCENARIOS_PATH = ROOT / "reports" / "business_value_scenarios.csv"
MLFLOW_SUMMARY_PATH = ROOT / "reports" / "mlflow_tracking_summary.csv"
MONGODB_EXPORT_PATH = ROOT / "reports" / "mongodb_export_summary.csv"

TARGET_COL = "Result"
PHISHING_LABEL = -1
LEGITIMATE_LABEL = 1
MODEL_PHISHING_LABELS = (-1, 0)

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REFERENCE_THRESHOLD = 0.50
DEFAULT_RECOMMENDED_THRESHOLD = 0.09
DEFAULT_BUSINESS_ASSUMPTIONS = {
    "cost_false_negative": 500.0,
    "cost_false_positive": 25.0,
    "cost_review": 2.0,
    "cost_operational_per_url": 0.50,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_business_summary():
    if not SUMMARY_PATH.exists():
        return {}, f"No se encontro {SUMMARY_PATH.relative_to(ROOT)}. Se usara umbral 0.09 por defecto."

    summary = pd.read_csv(SUMMARY_PATH)
    if not {"item", "value"}.issubset(summary.columns):
        return {}, "El resumen de Business Value no tiene columnas item/value. Se usaran valores por defecto."

    return dict(zip(summary["item"], summary["value"])), None


@st.cache_data
def load_business_scenarios():
    if not SCENARIOS_PATH.exists():
        return None, f"No se encontro {SCENARIOS_PATH.relative_to(ROOT)}. La seccion de escenarios se omitira."

    scenarios = pd.read_csv(SCENARIOS_PATH)
    if scenarios.empty:
        return None, "El archivo de escenarios esta vacio."

    return scenarios, None


@st.cache_data
def load_mlflow_summary():
    if not MLFLOW_SUMMARY_PATH.exists():
        return None, f"No se encontro {MLFLOW_SUMMARY_PATH.relative_to(ROOT)}."

    df = pd.read_csv(MLFLOW_SUMMARY_PATH)
    if df.empty:
        return df, "El resumen de MLflow existe, pero no contiene runs registrados."
    return df, None


@st.cache_data
def load_mongodb_export_summary():
    if not MONGODB_EXPORT_PATH.exists():
        return None, f"No se encontro {MONGODB_EXPORT_PATH.relative_to(ROOT)}."

    df = pd.read_csv(MONGODB_EXPORT_PATH)
    if df.empty:
        return df, "El resumen de exportacion MongoDB existe, pero no contiene registros."
    return df, None


def as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_thresholds(summary):
    recommended = as_float(
        summary.get("umbral_recomendado", summary.get("umbral_optimo")),
        DEFAULT_RECOMMENDED_THRESHOLD,
    )
    reference = as_float(summary.get("umbral_referencia"), DEFAULT_REFERENCE_THRESHOLD)
    return recommended, reference


def get_business_assumptions(scenarios):
    assumptions = DEFAULT_BUSINESS_ASSUMPTIONS.copy()
    if scenarios is None or "scenario" not in scenarios.columns:
        return assumptions

    base = scenarios[scenarios["scenario"].astype(str).str.lower() == "base"]
    if base.empty:
        return assumptions

    row = base.iloc[0]
    for key in assumptions:
        if key in row and pd.notna(row[key]):
            assumptions[key] = float(row[key])
    return assumptions


@st.cache_data(ttl=20)
def check_api_health(api_base_url):
    if not api_base_url:
        return "No configurada", "API_BASE_URL no configurada."

    url = api_base_url.rstrip("/") + "/health"
    try:
        response = requests.get(url, timeout=3)
        if response.ok:
            return "Online", f"Health check OK ({response.status_code})."
        return "Offline", f"Health check respondio {response.status_code}."
    except requests.RequestException as exc:
        return "Offline", str(exc)


def phishing_probability(model, X):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    phishing_model_label = next((label for label in MODEL_PHISHING_LABELS if label in classes), None)

    if phishing_model_label is None:
        raise ValueError(f"No se encontro clase phishing en el modelo: {classes}")

    return proba[:, classes.index(phishing_model_label)]


def predict_local(df, threshold):
    data = df.copy()
    if TARGET_COL in data.columns:
        data = data.drop(columns=[TARGET_COL])

    model = load_model()
    probabilities = phishing_probability(model, data)
    result = df.copy()
    result["phishing_probability"] = probabilities
    result["prediction"] = np.where(probabilities >= threshold, "phishing", "legitimo")
    return result


def parse_api_response(payload, original_df, threshold):
    if isinstance(payload, list):
        result = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        records = payload.get("results") or payload.get("data") or payload.get("predictions")
        if isinstance(records, list):
            result = pd.DataFrame(records)
        else:
            result = pd.DataFrame([payload])
    else:
        raise ValueError("Respuesta API no compatible.")

    if len(result) != len(original_df):
        raise ValueError("La API devolvio una cantidad de registros distinta al CSV enviado.")

    output = original_df.copy()
    for column in result.columns:
        output[column] = result[column].values

    if "phishing_probability" not in output.columns:
        for candidate in ["probability", "probability_phishing", "phishing_score", "score"]:
            if candidate in output.columns:
                output["phishing_probability"] = pd.to_numeric(output[candidate], errors="coerce")
                break

    if "prediction" not in output.columns and "phishing_probability" in output.columns:
        output["prediction"] = np.where(
            output["phishing_probability"].astype(float) >= threshold,
            "phishing",
            "legitimo",
        )

    if "prediction" not in output.columns or "phishing_probability" not in output.columns:
        raise ValueError("La respuesta API debe incluir prediction y phishing_probability, o una probabilidad equivalente.")

    return output


def predict_api(df, threshold, api_base_url):
    data = df.copy()
    if TARGET_COL in data.columns:
        data = data.drop(columns=[TARGET_COL])

    payload = {
        "threshold": threshold,
        "records": data.to_dict(orient="records"),
    }
    url = api_base_url.rstrip("/") + "/predict"
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    return parse_api_response(response.json(), df, threshold)


def confusion_or_expected_counts(result_df, threshold):
    probabilities = pd.to_numeric(result_df["phishing_probability"], errors="coerce").fillna(0).to_numpy()
    predicted_phishing = probabilities >= threshold

    if TARGET_COL in result_df.columns:
        y_true = result_df[TARGET_COL].to_numpy()
        actual_phishing = y_true == PHISHING_LABEL
        actual_legitimate = y_true == LEGITIMATE_LABEL
        return {
            "tn": int((actual_legitimate & ~predicted_phishing).sum()),
            "fp": int((actual_legitimate & predicted_phishing).sum()),
            "fn": int((actual_phishing & ~predicted_phishing).sum()),
            "tp": int((actual_phishing & predicted_phishing).sum()),
            "mode": "observado",
        }

    expected_tp = float(probabilities[predicted_phishing].sum())
    expected_fp = float((1 - probabilities[predicted_phishing]).sum())
    expected_fn = float(probabilities[~predicted_phishing].sum())
    expected_tn = float((1 - probabilities[~predicted_phishing]).sum())
    return {
        "tn": expected_tn,
        "fp": expected_fp,
        "fn": expected_fn,
        "tp": expected_tp,
        "mode": "estimado",
    }


def business_metrics(result_df, threshold, assumptions):
    counts = confusion_or_expected_counts(result_df, threshold)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total_urls = len(result_df)
    alerts = fp + tp
    phishing_real = fn + tp

    review_cost = alerts * assumptions["cost_review"]
    operational_cost = total_urls * assumptions["cost_operational_per_url"]
    operational_cost_total = review_cost + operational_cost
    cost_without_model = phishing_real * assumptions["cost_false_negative"]
    cost_with_model = (
        fn * assumptions["cost_false_negative"]
        + fp * assumptions["cost_false_positive"]
        + review_cost
        + operational_cost
    )
    net_savings = cost_without_model - cost_with_model
    roi = np.nan if operational_cost_total == 0 else net_savings / operational_cost_total
    value_per_1000_urls = np.nan if total_urls == 0 else net_savings / total_urls * 1000
    recall = np.nan if phishing_real == 0 else tp / phishing_real

    return {
        **counts,
        "threshold": threshold,
        "total_urls": total_urls,
        "alerts": alerts,
        "alert_rate": alerts / total_urls if total_urls else 0,
        "phishing_detected": alerts,
        "legitimate_sites": total_urls - alerts,
        "avg_phishing_probability": result_df["phishing_probability"].mean(),
        "recall": recall,
        "cost_without_model": cost_without_model,
        "cost_with_model": cost_with_model,
        "net_savings": net_savings,
        "roi": roi,
        "value_per_1000_urls": value_per_1000_urls,
    }


def format_money(value):
    return f"USD {value:,.2f}"


def format_count(value):
    if pd.isna(value):
        return "N/D"
    if abs(value - round(value)) < 0.001:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def render_mvp_status(mode, api_base_url, api_status, recommended_threshold):
    st.subheader("Estado del MVP Sprint 6")
    rows = [
        ("Modo activo", mode),
        ("API_BASE_URL", api_base_url or "No configurada"),
        ("Estado API", api_status),
        ("Modelo usado", "API REST / servicio remoto" if mode == "Modo API" else str(MODEL_PATH.relative_to(ROOT))),
        ("Umbral recomendado", f"{recommended_threshold:.2f}"),
        ("Fuente Business Value", str(SUMMARY_PATH.relative_to(ROOT)) if SUMMARY_PATH.exists() else "No disponible"),
        ("Fuente MLflow", str(MLFLOW_SUMMARY_PATH.relative_to(ROOT)) if MLFLOW_SUMMARY_PATH.exists() else "No disponible"),
        ("Fuente MongoDB", str(MONGODB_EXPORT_PATH.relative_to(ROOT)) if MONGODB_EXPORT_PATH.exists() else "No disponible"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Elemento", "Estado"]), use_container_width=True)


def render_mlflow_traceability():
    st.subheader("Trazabilidad experimental")
    mlflow_df, warning = load_mlflow_summary()
    if warning:
        st.warning(warning)
    if mlflow_df is None or mlflow_df.empty:
        return

    status_counts = mlflow_df["status"].value_counts() if "status" in mlflow_df.columns else pd.Series(dtype=int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs registrados", len(mlflow_df))
    c2.metric("Success", int(status_counts.get("success", 0)))
    c3.metric("Partial", int(status_counts.get("partial", 0)))
    c4.metric("Failed", int(status_counts.get("failed", 0)))

    columns = [
        col for col in ["model_name", "run_type", "sprint", "status", "metrics_logged"]
        if col in mlflow_df.columns
    ]
    table = mlflow_df[columns].copy()
    if "model_name" in table.columns:
        table.insert(0, "final_model", table["model_name"].astype(str).str.contains("final", case=False, na=False))
    st.dataframe(table, use_container_width=True)


def render_mongodb_status():
    st.subheader("Persistencia MongoDB")
    mongo_df, warning = load_mongodb_export_summary()
    if warning:
        st.warning(warning)
    if mongo_df is None or mongo_df.empty:
        return

    if "status" in mongo_df.columns:
        statuses = ", ".join(sorted(mongo_df["status"].dropna().astype(str).unique()))
        st.metric("Estado de exportacion", statuses or "N/D")
    if "notes" in mongo_df.columns and mongo_df["notes"].astype(str).str.contains("MONGODB_URI", na=False).any():
        st.info("La exportacion figura como skipped porque falta configurar MONGODB_URI. El dashboard no se conecta a MongoDB todavia.")

    st.dataframe(mongo_df, use_container_width=True)


summary, summary_warning = load_business_summary()
recommended_threshold, reference_threshold = get_thresholds(summary)
scenarios, scenarios_warning = load_business_scenarios()
assumptions = get_business_assumptions(scenarios)
default_api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)


st.title("Phishing Detection Platform")
st.caption("Panel MVP Sprint 6: dashboard local con preparacion para API REST")

if summary_warning:
    st.warning(summary_warning)

st.info(
    "Sprint 6 integra dashboard, API REST y despliegue. El modo local queda como respaldo "
    "operativo; el modo API sera el camino de despliegue cuando exista el servicio. "
    "Business Value y recall de phishing siguen siendo el centro de decision."
)


with st.sidebar:
    st.header("Configuracion")
    execution_mode = st.radio(
        "Modo de ejecucion",
        ["Modo local", "Modo API"],
        index=0,
    )
    api_base_url = st.text_input("API_BASE_URL", value=default_api_base_url)
    api_status, api_message = check_api_health(api_base_url)

    if api_status == "Online":
        st.success(f"API {api_status}")
    elif api_status == "Offline":
        st.warning(f"API {api_status}: {api_message}")
    else:
        st.info(api_status)

    st.metric("Umbral recomendado", f"{recommended_threshold:.2f}")
    st.metric("Umbral referencia", f"{reference_threshold:.2f}")
    threshold = st.slider(
        "Umbral de deteccion",
        min_value=0.01,
        max_value=0.99,
        value=float(recommended_threshold),
        step=0.01,
    )
    st.caption("Umbrales mas bajos priorizan recall de phishing; umbrales mas altos reducen falsas alarmas.")


render_mvp_status(execution_mode, api_base_url, api_status, recommended_threshold)
render_mlflow_traceability()
render_mongodb_status()


left, right = st.columns([1, 1])

with left:
    st.subheader("Carga de datos")
    uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])

with right:
    st.subheader("Estado")
    if uploaded_file is None:
        st.warning("Esperando archivo CSV.")
    else:
        st.success("Archivo cargado correctamente.")


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Vista previa")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("Procesando predicciones..."):
            if execution_mode == "Modo API":
                try:
                    result_df = predict_api(df=df, threshold=threshold, api_base_url=api_base_url)
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"No se pudo obtener prediccion desde la API: {exc}")
                    st.warning("Use modo local mientras la API REST de Sprint 6 no este disponible o estable.")
                    st.stop()
            else:
                result_df = predict_local(df=df, threshold=threshold)

        selected_metrics = business_metrics(result_df, threshold, assumptions)
        reference_metrics = business_metrics(result_df, reference_threshold, assumptions)
        recommended_metrics = business_metrics(result_df, recommended_threshold, assumptions)

        st.subheader("KPIs tecnicos")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total URLs evaluadas", format_count(selected_metrics["total_urls"]))
        k2.metric("Phishing detectados", format_count(selected_metrics["phishing_detected"]))
        k3.metric("Sitios legitimos", format_count(selected_metrics["legitimate_sites"]))
        k4.metric("Alertas generadas", format_count(selected_metrics["alerts"]))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Tasa de alertas", f"{selected_metrics['alert_rate']:.2%}")
        k6.metric("Prob. promedio phishing", f"{selected_metrics['avg_phishing_probability']:.2%}")
        k7.metric("Umbral usado", f"{selected_metrics['threshold']:.2f}")
        k8.metric("Modo de conteo", selected_metrics["mode"])

        st.subheader("KPIs economicos")
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("Costo sin modelo", format_money(selected_metrics["cost_without_model"]))
        e2.metric("Costo con modelo", format_money(selected_metrics["cost_with_model"]))
        e3.metric("Ahorro neto esperado", format_money(selected_metrics["net_savings"]))
        e4.metric("ROI", f"{selected_metrics['roi']:.2f}x")
        e5.metric("Valor por 1000 URLs", format_money(selected_metrics["value_per_1000_urls"]))
        e6.metric("Ahorro vs 0.50", format_money(selected_metrics["net_savings"] - reference_metrics["net_savings"]))

        st.subheader("Comparacion de umbrales")
        comparison = pd.DataFrame([
            {
                "escenario": f"Referencia {reference_threshold:.2f}",
                "threshold": reference_metrics["threshold"],
                "fp": reference_metrics["fp"],
                "fn": reference_metrics["fn"],
                "recall": reference_metrics["recall"],
                "net_savings": reference_metrics["net_savings"],
            },
            {
                "escenario": f"Recomendado {recommended_threshold:.2f}",
                "threshold": recommended_metrics["threshold"],
                "fp": recommended_metrics["fp"],
                "fn": recommended_metrics["fn"],
                "recall": recommended_metrics["recall"],
                "net_savings": recommended_metrics["net_savings"],
            },
        ])
        st.dataframe(comparison, use_container_width=True)

        d1, d2, d3 = st.columns(3)
        d1.metric("Diferencia FN", format_count(recommended_metrics["fn"] - reference_metrics["fn"]))
        d2.metric("Diferencia FP", format_count(recommended_metrics["fp"] - reference_metrics["fp"]))
        d3.metric(
            "Diferencia ahorro neto",
            format_money(recommended_metrics["net_savings"] - reference_metrics["net_savings"]),
        )

        st.subheader("Escenarios de sensibilidad")
        if scenarios_warning:
            st.warning(scenarios_warning)
        else:
            scenario_columns = [
                col for col in [
                    "scenario",
                    "recommended_threshold",
                    "recommended_recall",
                    "recommended_fp",
                    "recommended_fn",
                    "recommended_net_savings",
                    "annual_net_savings",
                ]
                if col in scenarios.columns
            ]
            st.dataframe(scenarios[scenario_columns], use_container_width=True)

        st.subheader("Resultados")
        st.dataframe(result_df.head(100), use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar resultados",
            data=csv,
            file_name="predicciones_phishing.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Error procesando archivo: {exc}")
