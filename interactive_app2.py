import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Config
# ============================================================
st.set_page_config(
    page_title="CVD 10-Year Risk – ML (GAN-Stacking Version)",
    layout="wide"
)

# Hard-coded evaluation summary from your Colab run
MODEL_SUMMARY = {
    "Logistic Regression": {"AUC": 0.860, "Brier": 0.104},
    "XGBoost":             {"AUC": 0.913, "Brier": 0.072},
    "Gradient Boosting":   {"AUC": 0.916, "Brier": 0.070},
    "Stacking":            {"AUC": 0.920, "Brier": 0.089},
}

# ============================================================
# Helpers
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found in repo root.")
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    return joblib.load(path)

def get_risk_category(p: float) -> tuple[str, str]:
    """
    Simple risk buckets based on 10-year CVD risk.
    """
    if p < 0.075:
        return "Low — Lifestyle focus", "green"
    elif p < 0.20:
        return "Moderate — Consider meds", "orange"
    else:
        return "High — Start treatment / refer", "red"

def format_percent(p: float) -> str:
    return f"{p*100:.1f}%"

def build_patient_row(age, sys_bp, dia_bp, chol, glu, bmi, smoke, fhx, note):
    return pd.DataFrame([{
        "note": note,
        "age": age,
        "sys_bp": sys_bp,
        "dia_bp": dia_bp,
        "cholesterol": chol,
        "glucose": glu,
        "bmi": bmi,
        "smoke": 1 if smoke == "Yes" else 0,
        "family_hx": 1 if fhx == "Yes" else 0,
    }])

def scenario_prediction(pipe, base_row: pd.DataFrame, **changes):
    row = base_row.copy()
    for k, v in changes.items():
        row.loc[:, k] = v
    p = float(pipe.predict_proba(row)[:, 1])
    return p

# ============================================================
# Load data + models
# ============================================================
DATA_PATH = "data_ehr_5000_gan_balanced.csv"

df = load_dataset(DATA_PATH)

# Pre-trained, calibrated models (they already include preprocessing)
LR_MODEL     = load_model("lr_calibrated.pkl")
XGB_MODEL    = load_model("xgb_calibrated.pkl")
GBM_MODEL    = load_model("gbm_calibrated.pkl")
STACK_MODEL  = load_model("stack_calibrated.pkl")

MODEL_MAP = {
    "Logistic Regression": LR_MODEL,
    "XGBoost":             XGB_MODEL,
    "Gradient Boosting":   GBM_MODEL,
    "Stacking":            STACK_MODEL,
}

# ============================================================
# Sidebar: Dataset + Model selection + Patient profile
# ============================================================
with st.sidebar:
    st.markdown("### Dataset status")
    prev = df["cvd"].mean()
    st.success(
        f"Rows: **{len(df):,}**  \n"
        f"Prevalence (CVD=1): **{prev*100:.1f}%**"
    )

    st.markdown("---")
    model_name = st.selectbox(
        "Model",
        ["Stacking", "XGBoost", "Gradient Boosting", "Logistic Regression"],
        index=0,
        help="All models are pre-trained and calibrated on a 5,000-record GAN-balanced dataset."
    )

    st.markdown("### Patient Profile")

    age = st.slider("Age", 20, 90, 55, step=1)
    sys_bp = st.slider("Systolic BP (mmHg)", 90, 220, 130, step=1)
    dia_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80, step=1)
    chol = st.slider("Cholesterol (mg/dL)", 120, 350, 190, step=1)
    glu = st.slider("Glucose (mg/dL)", 65, 260, 100, step=1)
    bmi = st.slider("BMI", 16.0, 50.0, 27.0, step=0.1)

    smoke = st.selectbox("Smoking", ["No", "Yes"])
    fhx = st.selectbox("Family History of CVD", ["No", "Yes"])

    note = st.text_area(
        "Clinical Note",
        value="no symptoms reported",
        height=110,
        help="Free-text summary: symptoms, lifestyle, adherence, etc."
    )

# ============================================================
# Main layout
# ============================================================
st.title("CVD 10-Year Risk – ML + GAN (Predictor v2)")
st.caption(
    "Demo app using a GAN-augmented 5,000-record synthetic EHR dataset and calibrated ML models. "
    "For educational purposes only – not a medical device."
)

left, right = st.columns([1.1, 1.6])

# ------------------------------------------------------------
# Build input row and run prediction
# ------------------------------------------------------------
row = build_patient_row(age, sys_bp, dia_bp, chol, glu, bmi, smoke, fhx, note)
model = MODEL_MAP[model_name]

with st.spinner("Computing risk..."):
    prob = float(model.predict_proba(row)[:, 1])

risk_label, risk_color = get_risk_category(prob)

# ------------------------------------------------------------
# LEFT: Input summary
# ------------------------------------------------------------
with left:
    st.subheader("Input Summary")

    st.markdown(
        f"""
        - **Age:** {age}  
        - **BP:** {sys_bp}/{dia_bp} mmHg  
        - **Cholesterol:** {chol} mg/dL  
        - **Glucose:** {glu} mg/dL  
        - **BMI:** {bmi:.1f}  
        - **Smoker:** {smoke}  
        - **Family Hx:** {fhx}  
        """
    )
    st.markdown("**Clinical Note (excerpt):**")
    st.write(f"_{note[:350]}{'…' if len(note) > 350 else ''}_")

    st.markdown("---")
    st.markdown("### Model Performance (on GAN-balanced dataset)")
    ms = MODEL_SUMMARY[model_name]
    st.markdown(
        f"- **Model:** {model_name}  \n"
        f"- **Test ROC AUC:** {ms['AUC']:.3f}  \n"
        f"- **Test Brier score:** {ms['Brier']:.3f}"
    )

# ------------------------------------------------------------
# RIGHT: Risk estimate + simple counterfactuals
# ------------------------------------------------------------
with right:
    st.subheader("CVD 10-Year Risk")

    st.markdown(
        f"<h2 style='color:{risk_color}; margin-bottom:0;'>{format_percent(prob)}</h2>",
        unsafe_allow_html=True
    )
    st.markdown(f"**{risk_label}**")

    st.caption(
        "These probabilities come from pre-trained, calibrated ML models (not a traditional score like Framingham). "
        "They reflect both structured data (age, BP, labs, BMI, smoking, family history) and the clinical note."
    )

    # ------------------------------
    # Simple counterfactuals
    # ------------------------------
    st.markdown("### What Lowers Risk Most? (Simple Scenarios)")

    scenarios = []

    # Lower SBP by 10
    sbp_lower = max(90, sys_bp - 10)
    p_sbp = scenario_prediction(model, row, sys_bp=sbp_lower)
    scenarios.append(("Lower SBP by 10 mmHg", p_sbp))

    # Lower DBP by 5
    dbp_lower = max(50, dia_bp - 5)
    p_dbp = scenario_prediction(model, row, dia_bp=dbp_lower)
    scenarios.append(("Lower DBP by 5 mmHg", p_dbp))

    # Lower cholesterol by 20
    chol_lower = max(120, chol - 20)
    p_chol = scenario_prediction(model, row, cholesterol=chol_lower)
    scenarios.append(("Lower cholesterol by 20 mg/dL", p_chol))

    # Lower BMI by 2
    bmi_lower = max(16.0, bmi - 2.0)
    p_bmi = scenario_prediction(model, row, bmi=bmi_lower)
    scenarios.append(("Lower BMI by 2 kg/m²", p_bmi))

    # Quit smoking
    if smoke == "Yes":
        p_qs = scenario_prediction(model, row, smoke=0)
        scenarios.append(("Quit smoking", p_qs))

    # Remove family history (hypothetical baseline)
    if fhx == "Yes":
        p_fh0 = scenario_prediction(model, row, family_hx=0)
        scenarios.append(("No family history (counterfactual)", p_fh0))

    if scenarios:
        rows = []
        for desc, p_cf in scenarios:
            abs_change = prob - p_cf
            pct_change = abs_change * 100
            rows.append({
                "Scenario": desc,
                "New risk": format_percent(p_cf),
                "Δ risk (points)": f"{abs_change*100:+.1f}",
            })
        cf_df = pd.DataFrame(rows)
        st.dataframe(cf_df, use_container_width=True, hide_index=True)
    else:
        st.write("No counterfactual scenarios defined for this profile.")

    # --------------------------------------------------------
    # Comparison vs simple 'traditional-like' baseline
    # --------------------------------------------------------
    st.markdown("### Comparison: Traditional-like vs ML")

    # A very rough 'traditional-like' risk based only on age & BP
    # (for illustration; not an actual guideline score)
    trad_logit = -7.0 + 0.04*(age-50) + 0.015*(sys_bp-120)
    trad_prob = 1 / (1 + np.exp(-trad_logit))
    st.markdown(
        f"- **Traditional-like score (illustrative):** {format_percent(trad_prob)}  \n"
        f"- **ML model ({model_name}) risk:** {format_percent(prob)}"
    )
    st.caption(
        "Traditional-like estimate uses only age and systolic BP. "
        "The ML model additionally incorporates diastolic BP, cholesterol, glucose, BMI, "
        "smoking, family history, and the clinical note."
    )

# ============================================================
# About this App + Disclaimer
# ============================================================
with st.expander("About This App"):
    st.markdown(
        """
        **What this is**

        - A demonstration of an AI-driven **10-year cardiovascular disease (CVD) risk predictor**.  
        - Uses a **synthetic, GAN-augmented dataset (5,000+ records)** that mimics real EHR data:
          age, blood pressure, cholesterol, glucose, BMI, smoking, family history, and a short clinical note.  
        - Models are **calibrated machine learning pipelines** (Logistic Regression, XGBoost, Gradient Boosting,
          and a Stacking ensemble).

        **How it works**

        1. A realistic synthetic cohort is generated.  
        2. A **GAN (Generative Adversarial Network)** is trained on CVD-positive cases to create
           additional minority samples and reach ≈30% prevalence.  
        3. On this GAN-balanced dataset, we train and calibrate the four models.  
        4. The app loads these pre-trained models and predicts CVD risk for a given patient profile and note.

        **Why a second app (v2)?**

        - Your first app (`cvd-predictor.streamlit.app`) is based on a smaller, more constrained dataset.  
        - This v2 app (`cvd-predictor2`) lets you **compare behavior** when models are trained on a
          larger, GAN-balanced synthetic EHR that better reflects real-world heterogeneity.
        """
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "© 2025 Howard Nguyen, PhD. Educational demo only — not for clinical decision-making."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem;'>"
    "This tool is built on synthetic, GAN-augmented data and calibrated ML models. "
    "It is designed for experimentation, validation, and stakeholder communication, "
    "not for direct patient care."
    "</p>",
    unsafe_allow_html=True,
)
