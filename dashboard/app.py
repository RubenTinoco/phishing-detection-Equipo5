import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.pkl"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
FINAL_COMPARISON_PATH = ROOT / "models" / "final_comparison.csv"
TARGET_COL = "Result"
PHISHING_LABEL = -1
LEGITIMATE_LABEL = 1
MODEL_PHISHING_LABELS = (-1, 0)


st.set_page_config(page_title="Sprint 5 - Phishing Detection", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def load_metrics():
    if FINAL_COMPARISON_PATH.exists():
        return pd.read_csv(FINAL_COMPARISON_PATH, index_col=0)
    return pd.DataFrame()


def phishing_probability(model, X):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    phishing_model_label = next((label for label in MODEL_PHISHING_LABELS if label in classes), None)
    if phishing_model_label is None:
        raise ValueError(f"No se pudo identificar la clase phishing en model.classes_: {classes}")
    return proba[:, classes.index(phishing_model_label)]


def predict_with_threshold(model, X, threshold):
    proba = phishing_probability(model, X)
    pred = pd.Series(
        [PHISHING_LABEL if p >= threshold else LEGITIMATE_LABEL for p in proba],
        index=X.index,
        name="prediction",
    )
    return pred, proba


def business_value(tn, fp, fn, tp, benefit_tp, cost_fp, cost_fn, benefit_tn):
    return tp * benefit_tp + fp * cost_fp + fn * cost_fn + tn * benefit_tn


st.title("Sprint 5 - Dashboard del Modelo Final")
st.caption("Deteccion de sitios web fraudulentos: KPIs, valor de negocio y simulador de predicciones.")

model = load_model()
test_df = load_test_data()
metrics_df = load_metrics()
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

with st.sidebar:
    st.header("Supuestos de negocio")
    threshold = st.slider("Umbral de phishing", 0.05, 0.95, 0.50, 0.01)
    benefit_tp = st.number_input("Beneficio por phishing detectado (TP)", value=100.0, step=10.0)
    cost_fp = st.number_input("Costo por sitio legitimo bloqueado (FP)", value=-20.0, step=5.0)
    cost_fn = st.number_input("Costo por phishing no detectado (FN)", value=-80.0, step=10.0)
    benefit_tn = st.number_input("Beneficio por legitimo permitido (TN)", value=0.0, step=5.0)
    annual_volume = st.number_input("Volumen anual estimado", value=100000, step=10000)

pred, proba = predict_with_threshold(model, X_test, threshold)
tn, fp, fn, tp = confusion_matrix(
    y_test,
    pred,
    labels=[LEGITIMATE_LABEL, PHISHING_LABEL],
).ravel()
value_test = business_value(tn, fp, fn, tp, benefit_tp, cost_fp, cost_fn, benefit_tn)
value_per_case = value_test / len(test_df)
annual_value = value_per_case * annual_volume

metric_cols = st.columns(5)
if not metrics_df.empty and "Modelo Final (Sprint 4)" in metrics_df.index:
    row = metrics_df.loc["Modelo Final (Sprint 4)"]
    metric_cols[0].metric("F1 Score", f"{row['F1']:.4f}")
    metric_cols[1].metric("AUC-ROC", f"{row['AUC-ROC']:.4f}")
    metric_cols[2].metric("Recall", f"{row['Recall']:.4f}")
    metric_cols[3].metric("Precision", f"{row['Precision']:.4f}")
else:
    metric_cols[0].metric("Test set", f"{len(test_df):,}")
metric_cols[4].metric("Valor anual estimado", f"USD {annual_value:,.0f}")

tab_summary, tab_predictions, tab_explain = st.tabs(["Resumen", "Predicciones", "Explicabilidad"])

with tab_summary:
    left, right = st.columns([1, 1])
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Real legitimo", "Real phishing"],
        columns=["Pred. legitimo", "Pred. phishing"],
    )
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
    fig_cm.update_layout(title="Matriz de confusion con umbral seleccionado")
    left.plotly_chart(fig_cm, use_container_width=True)

    value_df = pd.DataFrame({
        "Resultado": ["TN", "FP", "FN", "TP"],
        "Cantidad": [tn, fp, fn, tp],
        "Valor unitario": [benefit_tn, cost_fp, cost_fn, benefit_tp],
        "Valor total": [
            tn * benefit_tn,
            fp * cost_fp,
            fn * cost_fn,
            tp * benefit_tp,
        ],
    })
    right.dataframe(value_df, use_container_width=True, hide_index=True)
    right.metric("Valor esperado en test", f"USD {value_test:,.0f}")
    right.metric("Valor esperado por URL", f"USD {value_per_case:,.2f}")

with tab_predictions:
    uploaded = st.file_uploader("Subir CSV para simular predicciones", type="csv")
    input_df = pd.read_csv(uploaded) if uploaded else X_test.copy()
    if TARGET_COL in input_df.columns:
        input_features = input_df.drop(columns=[TARGET_COL])
    else:
        input_features = input_df

    pred_uploaded, proba_uploaded = predict_with_threshold(model, input_features, threshold)
    result_df = input_df.copy()
    result_df["phishing_probability"] = proba_uploaded
    result_df["prediction"] = pred_uploaded.map({PHISHING_LABEL: "phishing", LEGITIMATE_LABEL: "legitimo"})
    st.dataframe(result_df.head(100), use_container_width=True)

with tab_explain:
    st.subheader("Importancia de variables")
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        fig_imp = px.bar(importance, x="importance", y="feature", orientation="h")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("El modelo no expone feature_importances_. Para SHAP, instalar shap y agregar el explainer correspondiente.")

    idx = st.slider("Instancia del test set", 0, len(X_test) - 1, 0)
    st.write("Probabilidad de phishing:", f"{proba[idx]:.3f}")
    st.dataframe(X_test.iloc[[idx]], use_container_width=True)
