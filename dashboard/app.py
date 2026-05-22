# app.py

from pathlib import Path
import tempfile

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURACION
# =========================================================

st.set_page_config(
    page_title="Phishing Detection Platform",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("models/final_model.pkl")

TARGET_COL = "Result"
PHISHING_LABEL = -1
LEGITIMATE_LABEL = 1
MODEL_PHISHING_LABELS = (-1, 0)


# =========================================================
# CARGAR MODELO
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# FUNCIONES
# =========================================================

def phishing_probability(model, X):
    proba = model.predict_proba(X)

    classes = list(model.classes_)

    phishing_model_label = next(
        (label for label in MODEL_PHISHING_LABELS if label in classes),
        None
    )

    if phishing_model_label is None:
        raise ValueError(
            f"No se encontro clase phishing en: {classes}"
        )

    return proba[:, classes.index(phishing_model_label)]


def predict_pipeline(df, threshold=0.5):

    # =====================================================
    # LIMPIEZA BASICA
    # =====================================================

    data = df.copy()

    if TARGET_COL in data.columns:
        data = data.drop(columns=[TARGET_COL])

    # =====================================================
    # PREDICCIONES
    # =====================================================

    probabilities = phishing_probability(model, data)

    predictions = [
        "phishing" if p >= threshold else "legitimo"
        for p in probabilities
    ]

    # =====================================================
    # RESULTADOS
    # =====================================================

    result = df.copy()

    result["phishing_probability"] = probabilities
    result["prediction"] = predictions

    return result


# =========================================================
# HEADER
# =========================================================

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        font-size: 50px;
        font-weight: 700;
        color: white;
    }

    .subtitle {
        font-size: 20px;
        color: #A0A0A0;
    }

    .card {
        background-color: #161B22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363D;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="title">
        🛡️ Phishing Detection Platform
    </div>

    <div class="subtitle">
        Plataforma inteligente para detección automatizada de URLs fraudulentas
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.write("")


# =========================================================
# SIDEBAR
# =========================================================

#with st.sidebar:

#    st.header("⚙️ Configuracion")

#    threshold = st.slider(
#        "Umbral de deteccion",
#        min_value=0.05,
 #       max_value=0.95,
 #       value=0.50,
 #       step=0.01
 #   )

#    st.info(
#        """
#        Mientras mas alto el umbral:
#        
#        - menos falsos positivos
#        - mas estricta la deteccion
#        """
#)
# =========================================================
# CONFIGURACION FIJA
# =========================================================

threshold = 0.50

# =========================================================
# LAYOUT
# =========================================================

left, right = st.columns([1, 1])


# =========================================================
# SUBIDA CSV
# =========================================================

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📂 Subir archivo CSV")

    uploaded_file = st.file_uploader(
        "Seleccione un archivo",
        type=["csv"]
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# RESULTADOS
# =========================================================

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Estado")

    if uploaded_file is not None:
        st.success("Archivo cargado correctamente")
    else:
        st.warning("Esperando archivo CSV")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PROCESAMIENTO
# =========================================================

if uploaded_file is not None:

    try:

        # =================================================
        # LEER CSV
        # =================================================

        df = pd.read_csv(uploaded_file)

        st.write("")
        st.subheader("🔎 Vista previa")

        st.dataframe(df.head(), use_container_width=True)

        # =================================================
        # PIPELINE
        # =================================================

        with st.spinner("Procesando predicciones..."):

            result_df = predict_pipeline(
                df=df,
                threshold=threshold
            )

        # =================================================
        # KPIS
        # =================================================

        phishing_count = (
            result_df["prediction"] == "phishing"
        ).sum()

        legit_count = (
            result_df["prediction"] == "legitimo"
        ).sum()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total registros",
            len(result_df)
        )

        c2.metric(
            "Phishing detectados",
            phishing_count
        )

        c3.metric(
            "Sitios legitimos",
            legit_count
        )

        # =================================================
        # RESULTADOS
        # =================================================

        st.write("")
        st.subheader("📄 Resultados")

        st.dataframe(
            result_df.head(100),
            use_container_width=True
        )

        # =================================================
        # DESCARGA
        # =================================================

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Descargar resultados",
            data=csv,
            file_name="predicciones_phishing.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Error procesando archivo: {str(e)}")
