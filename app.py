"""
Diabetes Prediction Web Application
====================================
Streamlit frontend that loads the trained MLP model weights (numpy)
and scaler, and provides both custom input and random sample prediction.

No TensorFlow dependency — uses pure NumPy for inference.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction — MLP",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Main background ─────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #1a1040 40%, #0f172a 100%);
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.97) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #94a3b8;
    }

    /* ── Headers ─────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Metric cards ────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,.25), 0 0 30px rgba(99,102,241,.06);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,.3) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Number inputs ───────────────────────────────────── */
    .stNumberInput > div > div > input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        transition: all 0.3s ease !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,.18) !important;
    }

    /* ── Result boxes ────────────────────────────────────── */
    .result-positive {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        animation: fadeSlideIn 0.5s ease-out;
    }
    .result-negative {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        animation: fadeSlideIn 0.5s ease-out;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(15px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Info cards row ───────────────────────────────────── */
    .info-card {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 14px;
        padding: 24px;
        transition: all 0.3s ease;
        height: 100%;
    }
    .info-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,.2);
    }

    /* ── Tabs ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 10px 24px !important;
        background: rgba(17, 24, 39, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.08) !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #a78bfa !important;
        font-weight: 600 !important;
    }

    /* ── Data frames ─────────────────────────────────────── */
    .stDataFrame {
        border-radius: 14px !important;
        overflow: hidden;
    }

    /* ── Dividers ─────────────────────────────────────────── */
    hr {
        border-color: rgba(99, 102, 241, 0.1) !important;
    }

    /* ── Badge ────────────────────────────────────────────── */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99,102,241,.1);
        border: 1px solid rgba(99,102,241,.2);
        border-radius: 50px;
        padding: 8px 20px;
        font-size: .85rem;
        font-weight: 500;
        color: #818cf8;
        letter-spacing: 0.4px;
    }
    .hero-badge .pulse-dot {
        width: 8px; height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: pulseDot 2s infinite;
    }
    @keyframes pulseDot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,.4); }
        50%      { opacity: .7; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    }

    /* ── Gradient title ───────────────────────────────────── */
    .gradient-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin: 20px 0 10px;
    }

    /* ── Scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0e17; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,.5); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load Artefacts (cached so they load only once)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.npz")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

FEATURE_INFO = {
    "Pregnancies":              {"min": 0,    "max": 17,    "desc": "Number of pregnancies",                 "step": 1,     "format": "%d"},
    "Glucose":                  {"min": 0.0,  "max": 200.0, "desc": "Plasma glucose concentration (mg/dL)",  "step": 1.0,   "format": "%.0f"},
    "BloodPressure":            {"min": 0.0,  "max": 140.0, "desc": "Diastolic blood pressure (mm Hg)",      "step": 1.0,   "format": "%.0f"},
    "SkinThickness":            {"min": 0.0,  "max": 100.0, "desc": "Triceps skinfold thickness (mm)",       "step": 1.0,   "format": "%.0f"},
    "Insulin":                  {"min": 0.0,  "max": 900.0, "desc": "2-Hour serum insulin (μU/mL)",          "step": 1.0,   "format": "%.0f"},
    "BMI":                      {"min": 0.0,  "max": 70.0,  "desc": "Body mass index (kg/m²)",               "step": 0.1,   "format": "%.1f"},
    "DiabetesPedigreeFunction": {"min": 0.0,  "max": 2.5,   "desc": "Diabetes pedigree function score",      "step": 0.001, "format": "%.3f"},
    "Age":                      {"min": 18,   "max": 100,   "desc": "Age in years",                          "step": 1,     "format": "%d"},
}


@st.cache_resource(show_spinner="🧠 Loading model weights …")
def load_weights():
    """Load model weights from .npz file."""
    return dict(np.load(WEIGHTS_PATH))


@st.cache_resource(show_spinner="⚙️ Loading scaler …")
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data(show_spinner="📊 Loading dataset …")
def load_data():
    return pd.read_csv(DATA_URL)


weights = load_weights()
scaler = load_scaler()
df = load_data()


# ---------------------------------------------------------------------------
# Pure NumPy MLP Inference
# ---------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def predict_diabetes(values: list[float]) -> dict:
    """
    Scale input, run MLP inference (pure NumPy), return prediction details.
    Architecture: Input(8) → Dense(32,ReLU) → Dense(16,ReLU) → Dense(8,ReLU) → Dense(1,Sigmoid)
    """
    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    # Forward pass (dropout is NOT applied at inference time)
    x = arr_scaled
    x = relu(x @ weights['w1'] + weights['b1'])   # Dense 32
    x = relu(x @ weights['w2'] + weights['b2'])   # Dense 16
    x = relu(x @ weights['w3'] + weights['b3'])   # Dense 8
    x = sigmoid(x @ weights['w4'] + weights['b4'])  # Dense 1

    prob = float(x[0][0])
    prediction = int(prob > 0.5)

    return {
        "prediction": prediction,
        "label": "Diabetic" if prediction == 1 else "Not Diabetic",
        "confidence": round(prob * 100 if prediction == 1 else (1 - prob) * 100, 2),
        "probability": round(prob * 100, 2),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 About This App")
    st.markdown(
        "This application uses a **Multilayer Perceptron (MLP)** "
        "neural network trained on the **PIMA Indian Diabetes Dataset** "
        "to predict whether a patient is likely to have diabetes."
    )

    st.divider()

    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- **Samples:** {len(df)}")
    st.markdown(f"- **Features:** {len(FEATURE_NAMES)}")
    diabetic_count = int(df['Outcome'].sum())
    non_diabetic = len(df) - diabetic_count
    st.markdown(f"- **Diabetic:** {diabetic_count}  ({diabetic_count / len(df) * 100:.1f}%)")
    st.markdown(f"- **Non-Diabetic:** {non_diabetic}  ({non_diabetic / len(df) * 100:.1f}%)")

    st.divider()

    st.markdown("### 🧠 Model Architecture")
    st.code(
        "Input (8 features)\n"
        "  → Dense(32, ReLU)\n"
        "  → Dropout(0.3)\n"
        "  → Dense(16, ReLU)\n"
        "  → Dropout(0.2)\n"
        "  → Dense(8, ReLU)\n"
        "  → Dense(1, Sigmoid)\n"
        "\n"
        "Optimizer : Adam (lr=0.001)\n"
        "Loss      : Binary Cross-Entropy",
        language=None,
    )

    st.divider()

    st.markdown("### ⚙️ Tech Stack")
    st.markdown(
        "- **Frontend:** Streamlit\n"
        "- **Inference:** NumPy (pure)\n"
        "- **Scaler:** scikit-learn StandardScaler\n"
        "- **Original Training:** TensorFlow / Keras"
    )

    st.divider()
    st.caption("Built with ❤️ using Streamlit • PIMA Indian Diabetes Dataset")


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='text-align:center; padding: 10px 0 5px;'>"
    "<span class='hero-badge'>"
    "<span class='pulse-dot'></span>"
    "Model Loaded &amp; Ready"
    "</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align:center; padding: 0 0 30px;'>"
    "<div class='gradient-title'>Diabetes Prediction</div>"
    "<p style='color:#94a3b8; font-size:1.1rem; max-width:600px; margin:0 auto;'>"
    "Predict diabetes risk using a trained Multilayer Perceptron neural network"
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_custom, tab_random, tab_explore = st.tabs([
    "🔬  Custom Input",
    "🎲  Random Sample",
    "📋  Explore Dataset",
])


# =====================  TAB 1 — Custom Input  ============================
with tab_custom:
    st.markdown("#### Enter Patient Health Parameters")
    st.caption("Fill in the 8 features below and click **Predict** to get the diagnosis prediction.")

    st.markdown("")

    col1, col2 = st.columns(2, gap="large")
    input_values = {}

    for idx, feat in enumerate(FEATURE_NAMES):
        info = FEATURE_INFO[feat]
        target_col = col1 if idx < 4 else col2
        with target_col:
            input_values[feat] = st.number_input(
                f"**{feat}**",
                min_value=info["min"],
                max_value=info["max"],
                value=info["min"],
                step=info["step"],
                format=info["format"],
                help=info["desc"],
                key=f"custom_{feat}",
            )

    st.markdown("")

    btn_col1, btn_col2, _ = st.columns([1, 1, 3])

    with btn_col1:
        predict_clicked = st.button(
            "🔮  Predict", type="primary", use_container_width=True, key="btn_predict_custom"
        )
    with btn_col2:
        clear_clicked = st.button(
            "🗑️  Clear", use_container_width=True, key="btn_clear_custom"
        )

    if clear_clicked:
        st.rerun()

    if predict_clicked:
        values = [float(input_values[f]) for f in FEATURE_NAMES]

        with st.spinner("Running prediction …"):
            result = predict_diabetes(values)

        st.markdown("---")
        st.markdown("#### 📊 Prediction Result")

        # Big result card
        if result["prediction"] == 1:
            st.markdown(
                "<div class='result-positive'>"
                "<div style='font-size:3.2rem;'>⚠️</div>"
                "<div style='font-size:1.8rem; font-weight:800; color:#ef4444; margin:12px 0 6px;'>"
                f"{result['label']}</div>"
                "<div style='color:#94a3b8; font-size:.95rem;'>"
                "The model predicts this patient is <strong style='color:#f87171;'>likely diabetic</strong>."
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-negative'>"
                "<div style='font-size:3.2rem;'>✅</div>"
                "<div style='font-size:1.8rem; font-weight:800; color:#10b981; margin:12px 0 6px;'>"
                f"{result['label']}</div>"
                "<div style='color:#94a3b8; font-size:.95rem;'>"
                "The model predicts this patient is <strong style='color:#34d399;'>not diabetic</strong>."
                "</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("🏷️ Prediction", result["label"])
        m2.metric("🎯 Confidence", f"{result['confidence']}%")
        m3.metric("📈 Diabetes Prob.", f"{result['probability']}%")

        # Confidence bar
        st.markdown("")
        st.markdown("**Confidence Level**")
        st.progress(min(int(result["confidence"]), 100))


# =====================  TAB 2 — Random Sample  ===========================
with tab_random:
    st.markdown("#### 🎲 Random Sample from Dataset")
    st.caption(
        "Click the button below to pick a **random patient** from the PIMA dataset, "
        "auto-fill the values, run the prediction, and compare with the **actual outcome**."
    )

    st.markdown("")

    if st.button("🎯  Pick Random Sample & Predict", type="primary", key="btn_random"):
        # Pick random row
        sample_row = df.sample(1).iloc[0]
        actual_outcome = int(sample_row["Outcome"])
        actual_label = "Diabetic" if actual_outcome == 1 else "Not Diabetic"

        values = [float(sample_row[f]) for f in FEATURE_NAMES]

        with st.spinner("Running prediction …"):
            result = predict_diabetes(values)

        # Show sampled data
        st.markdown("---")
        st.markdown("#### 📝 Sampled Patient Data")
        sample_display = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Value": values,
            "Description": [FEATURE_INFO[f]["desc"] for f in FEATURE_NAMES],
        })
        st.dataframe(sample_display, use_container_width=True, hide_index=True)

        st.markdown("")

        # Prediction vs Actual — side by side
        st.markdown("#### 🤖 Prediction vs 🏥 Actual")

        r1, r2 = st.columns(2, gap="large")

        with r1:
            pred_cls = "result-positive" if result["prediction"] == 1 else "result-negative"
            pred_color = "#ef4444" if result["prediction"] == 1 else "#10b981"
            st.markdown(
                f"<div class='{pred_cls}'>"
                "<div style='font-size:2.2rem;'>🤖</div>"
                "<div style='font-size:.75rem; color:#64748b; text-transform:uppercase; "
                "letter-spacing:1.2px; margin:10px 0 4px; font-weight:600;'>Model Prediction</div>"
                f"<div style='font-size:1.5rem; font-weight:800; color:{pred_color};'>"
                f"{result['label']}</div>"
                f"<div style='color:#94a3b8; font-size:.88rem; margin-top:8px;'>"
                f"Confidence: <strong>{result['confidence']}%</strong></div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with r2:
            actual_cls = "result-positive" if actual_outcome == 1 else "result-negative"
            actual_color = "#ef4444" if actual_outcome == 1 else "#10b981"
            st.markdown(
                f"<div class='{actual_cls}'>"
                "<div style='font-size:2.2rem;'>🏥</div>"
                "<div style='font-size:.75rem; color:#64748b; text-transform:uppercase; "
                "letter-spacing:1.2px; margin:10px 0 4px; font-weight:600;'>Actual Outcome</div>"
                f"<div style='font-size:1.5rem; font-weight:800; color:{actual_color};'>"
                f"{actual_label}</div>"
                "<div style='color:#94a3b8; font-size:.88rem; margin-top:8px;'>"
                "Ground truth from dataset</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Match/mismatch indicator
        if result["prediction"] == actual_outcome:
            st.success("✅  **Correct!** The model's prediction matches the actual outcome.", icon="🎯")
        else:
            st.error("❌  **Mismatch!** The model's prediction differs from the actual outcome.", icon="⚡")

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("🏷️ Prediction", result["label"])
        m2.metric("📈 Diabetes Prob.", f"{result['probability']}%")
        m3.metric("🏥 Actual", actual_label)


# =====================  TAB 3 — Explore Dataset  =========================
with tab_explore:
    st.markdown("#### 📋 PIMA Indian Diabetes Dataset")
    st.caption(f"Showing all **{len(df)}** samples from the dataset.")

    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("---")

    st.markdown("#### 📈 Statistical Summary")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.markdown("---")

    st.markdown("#### 📖 Feature Reference Guide")
    ref_data = []
    for feat in FEATURE_NAMES:
        info = FEATURE_INFO[feat]
        ref_data.append({
            "Feature": feat,
            "Description": info["desc"],
            "Dataset Min": round(df[feat].min(), 2),
            "Dataset Max": round(df[feat].max(), 2),
            "Dataset Mean": round(df[feat].mean(), 2),
            "Dataset Std": round(df[feat].std(), 2),
        })
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("#### 🎯 Class Distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Non-Diabetic (0)", f"{non_diabetic} ({non_diabetic / len(df) * 100:.1f}%)")
    with c2:
        st.metric("Diabetic (1)", f"{diabetic_count} ({diabetic_count / len(df) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:.82rem; padding:10px 0 30px;'>"
    "🩺 Diabetes Prediction using Multilayer Perceptrons &nbsp;•&nbsp; "
    "Built with Streamlit &amp; NumPy &nbsp;•&nbsp; "
    "Dataset: PIMA Indian Diabetes"
    "</div>",
    unsafe_allow_html=True,
)
