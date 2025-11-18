import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="CVD 10-Year Risk – ML + GAN (Predictor v2)",
    layout="wide"
)

# Hard-coded evaluation summary from Colab
MODEL_SUMMARY = {
    "Logistic Regression": {"AUC": 0.860, "Brier": 0.104},
    "XGBoost":             {"AUC": 0.913, "Brier": 0.072},
    "Gradient Boosting":   {"AUC": 0.916, "Brier": 0.070},
    "Stacking":            {"AUC": 0.920, "Brier": 0.089},
}

# Anchor clinical notes for comparison
POSITIVE_NOTE = (
    "active lifestyle, healthy diet, exercises regularly, no symptoms, "
    "never smoked, good sleep, no cardiac complaints"
)

NEGATIVE_NOTE = (
    "chest pain, palpitations, shortness of breath, dizziness on exertion, "
    "fatigue, noncompliant with medications, poor diet, high stress"
)

# ============================================================
# Helper functions
# ============================================================
def get_risk_category(p: float):
    """Return (label, color) based on 10-year CVD risk."""
    if p < 0.075:
        return "Low — Lifestyle focus", "green"
    elif p < 0.20:
        return "Moderate — Consider meds", "orange"
    else:
        return "High — Start treatment / refer", "red"


def format_percent(p: float) -> str:
    return f"{p * 100:.1f}%"


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
    """Clone base_row, apply changes, predict prob."""
    row = base_row.copy()
    for k, v in changes.items():
        row.loc[:, k] = v
    p = float(pipe.predict_proba(row)[:, 1])
    return p


def local_jitter_uncertainty(model, base_row: pd.DataFrame, n_samples: int = 200):
    """
    Estimate local prediction uncertainty by jittering numeric inputs slightly,
    then returning 5th, 50th, 95th percentiles of predicted risk.
    """
    rows = pd.concat([base_row] * n_samples, ignore_index=True)

    # Jitter numeric fields with small Gaussian noise
    rng = np.random.default_rng(42)
    rows["age"] = np.clip(
        rows["age"] + rng.normal(0, 1, n_samples), 20, 90
    ).round().astype(int)
    rows["sys_bp"] = np.clip(
        rows["sys_bp"] + rng.normal(0, 5, n_samples), 90, 220
    ).round().astype(int)
    rows["dia_bp"] = np.clip(
        rows["dia_bp"] + rng.normal(0, 4, n_samples), 50, 130
    ).round().astype(int)
    rows["cholesterol"] = np.clip(
        rows["cholesterol"] + rng.normal(0, 10, n_samples), 120, 350
    ).round().astype(int)
    rows["glucose"] = np.clip(
        rows["glucose"] + rng.normal(0, 10, n_samples), 65, 260
    ).round().astype(int)
    rows["bmi"] = np.clip(
        rows["bmi"] + rng.normal(0, 0.6, n_samples), 16.0, 50.0
    )

    probs = model.predict_proba(rows)[:, 1]
    low, med, high = np.percentile(probs, [5, 50, 95])
    return float(low), float(med), float(high)


def analyze_note_keywords(note: str):
    """
    Very simple NLP panel: mark 'risky' vs 'protective' phrases in the note.
    """
    note_l = note.lower()

    risk_terms = [
        "chest pain",
        "chest pressure",
        "shortness of breath",
        "palpitations",
        "dizziness",
        "fatigue on exertion",
        "sedentary",
        "poor diet",
        "high stress",
        "noncompliant",
        "smokes daily",
    ]
    protect_terms = [
        "active lifestyle",
        "healthy diet",
        "no symptoms reported",
        "never smoked",
        "quit smoking",
        "exercises regularly",
    ]

    risk_hits = [t for t in risk_terms if t in note_l]
    protect_hits = [t for t in protect_terms if t in note_l]

    return risk_hits, protect_hits


def get_percentile(prob: float, dist: np.ndarray) -> float:
    """Return percentile (0–1) of prob within distribution dist."""
    return float((dist <= prob).mean())


# ============================================================
# Caching: dataset + model load (safe, hashable)
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found in repo root.")
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    return joblib.load(path)


# ============================================================
# Load data + models
# ============================================================
DATA_PATH = "data_ehr_5000_gan_balanced.csv"
df = load_dataset(DATA_PATH)

LR_MODEL = load_model("lr_calibrated.pkl")
XGB_MODEL = load_model("xgb_calibrated.pkl")
GBM_MODEL = load_model("gbm_calibrated.pkl")
STACK_MODEL = load_model("stack_calibrated.pkl")

MODEL_MAP = {
    "Logistic Regression": LR_MODEL,
    "XGBoost": XGB_MODEL,
    "Gradient Boosting": GBM_MODEL,
    "Stacking": STACK_MODEL,
}


def precompute_risk_distributions(df_full: pd.DataFrame):
    """
    Compute predicted risk for all rows (no cvd label) for each model,
    for use in percentile context. No caching decorator here because
    it depends on model objects, which are not hashable.
    """
    X_all = df_full.drop(columns=["cvd"])
    dists = {}
    dists["Logistic Regression"] = LR_MODEL.predict_proba(X_all)[:, 1]
    dists["XGBoost"] = XGB_MODEL.predict_proba(X_all)[:, 1]
    dists["Gradient Boosting"] = GBM_MODEL.predict_proba(X_all)[:, 1]
    dists["Stacking"] = STACK_MODEL.predict_proba(X_all)[:, 1]
    return dists


# Precompute once on import
RISK_DISTS = precompute_risk_distributions(df)

# ============================================================
# Sidebar: dataset + model + patient profile
# ============================================================
with st.sidebar:
    st.markdown("### Dataset status")
    prevalence = df["cvd"].mean()
    st.success(
        f"Rows: **{len(df):,}**  \n"
        f"Prevalence (CVD=1): **{prevalence*100:.1f}%**"
    )

    st.markdown("---")
    model_name = st.selectbox(
        "Model",
        ["Stacking", "XGBoost", "Gradient Boosting", "Logistic Regression"],
        index=0,
        help="Calibrated ML models trained on a 5,000-record GAN-balanced synthetic EHR dataset.",
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
        help="Free-text summary: symptoms, lifestyle, adherence, etc.",
    )

# ============================================================
# Main layout
# ============================================================
st.title("CVD 10-Year Risk – ML + GAN (Predictor v2)")
st.caption(
    "Demo app using a GAN-augmented 5,000-record synthetic EHR dataset and calibrated ML models. "
    "For educational purposes only – not a medical device."
)

left, right = st.columns([1.2, 1.8])

# Build input row and run prediction
row = build_patient_row(age, sys_bp, dia_bp, chol, glu, bmi, smoke, fhx, note)
model = MODEL_MAP[model_name]

with st.spinner("Computing risk and local uncertainty..."):
    prob = float(model.predict_proba(row)[:, 1])

    # Local jitter-based uncertainty (5–95% band)
    low_u, med_u, high_u = local_jitter_uncertainty(model, row, n_samples=200)

    # Percentile vs population distribution
    dist = RISK_DISTS[model_name]
    percentile = get_percentile(prob, dist)

risk_label, risk_color = get_risk_category(prob)

# ============================================================
# LEFT: Input summary + performance
# ============================================================
with left:
    st.subheader("Input Summary")

    st.markdown(
        f"- **Age:** {age}  \n"
        f"- **BP:** {sys_bp}/{dia_bp} mmHg  \n"
        f"- **Cholesterol:** {chol} mg/dL  \n"
        f"- **Glucose:** {glu} mg/dL  \n"
        f"- **BMI:** {bmi:.1f}  \n"
        f"- **Smoker:** {smoke}  \n"
        f"- **Family Hx:** {fhx}"
    )

    st.markdown("**Clinical Note (excerpt):**")
    st.write(f"_{note[:350]}{'…' if len(note) > 350 else ''}_")

    st.markdown("---")
    st.markdown("### Model Performance (GAN-balanced dataset)")
    ms = MODEL_SUMMARY[model_name]
    st.markdown(
        f"- **Model:** {model_name}  \n"
        f"- **Test ROC AUC:** {ms['AUC']:.3f}  \n"
        f"- **Test Brier score:** {ms['Brier']:.3f}  \n"
        f"- **Population prevalence (training):** {prevalence*100:.1f}%"
    )

    st.markdown("---")
    st.markdown("### Risk Context in Population")
    st.markdown(
        f"- **Your predicted risk:** {format_percent(prob)}  \n"
        f"- **Population percentile:** ~**{percentile*100:.0f}th** percentile "
        "(higher than this share of the synthetic cohort)."
    )

# ============================================================
# RIGHT: Risk, uncertainty, counterfactuals, text impact
# ============================================================
with right:
    st.subheader("CVD 10-Year Risk")

    st.markdown(
        f"<h2 style='color:{risk_color}; margin-bottom:0;'>{format_percent(prob)}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{risk_label}**")

    st.caption(
        "Risk is estimated from a pre-trained, calibrated ML model. "
        "Inputs include age, BP, cholesterol, glucose, BMI, smoking, family history, and the clinical note."
    )

    # Local uncertainty band
    st.markdown("#### Model Uncertainty (Local)")
    st.markdown(
        f"- **Jittered risk band (5–95%):** {format_percent(low_u)} – {format_percent(high_u)}  \n"
        f"- **Median under jitter:** {format_percent(med_u)}"
    )
    st.caption(
        "Uncertainty is approximated by slightly perturbing numeric inputs (BP, labs, BMI, age) "
        "and recomputing risk 200 times. It reflects local stability of the prediction, "
        "not dataset-wide confidence intervals."
    )

    # ------------------------------
    # Counterfactual scenarios
    # ------------------------------
    st.markdown("### What Lowers Risk Most? (Simple Scenarios)")

    scenarios = []

    sbp_lower = max(90, sys_bp - 10)
    p_sbp = scenario_prediction(model, row, sys_bp=sbp_lower)
    scenarios.append(("Lower SBP by 10 mmHg", p_sbp))

    dbp_lower = max(50, dia_bp - 5)
    p_dbp = scenario_prediction(model, row, dia_bp=dbp_lower)
    scenarios.append(("Lower DBP by 5 mmHg", p_dbp))

    chol_lower = max(120, chol - 20)
    p_chol = scenario_prediction(model, row, cholesterol=chol_lower)
    scenarios.append(("Lower cholesterol by 20 mg/dL", p_chol))

    bmi_lower = max(16.0, bmi - 2.0)
    p_bmi = scenario_prediction(model, row, bmi=bmi_lower)
    scenarios.append(("Lower BMI by 2 kg/m²", p_bmi))

    if smoke == "Yes":
        p_qs = scenario_prediction(model, row, smoke=0)
        scenarios.append(("Quit smoking", p_qs))

    if fhx == "Yes":
        p_fh0 = scenario_prediction(model, row, family_hx=0)
        scenarios.append(("No family history (counterfactual)", p_fh0))

    if scenarios:
        rows_cf = []
        for desc, p_cf in scenarios:
            abs_change = prob - p_cf
            rows_cf.append({
                "Scenario": desc,
                "New risk": format_percent(p_cf),
                "Δ risk (points)": f"{abs_change*100:+.1f}",
            })
        cf_df = pd.DataFrame(rows_cf)
        st.dataframe(cf_df, use_container_width=True, hide_index=True)
    else:
        st.write("No counterfactual scenarios defined for this profile.")

    # --------------------------------------------------------
    # Clinical Note impact and keywords
    # --------------------------------------------------------
    st.markdown("### Clinical Note Impact")

    # Your note
    prob_your = prob

    # Positive anchor note
    row_pos = row.copy()
    row_pos.loc[:, "note"] = POSITIVE_NOTE
    prob_pos = float(model.predict_proba(row_pos)[:, 1])

    # Neutral note
    row_neu = row.copy()
    row_neu.loc[:, "note"] = "no symptoms reported"
    prob_neu = float(model.predict_proba(row_neu)[:, 1])

    # High-risk anchor note
    row_neg = row.copy()
    row_neg.loc[:, "note"] = NEGATIVE_NOTE
    prob_neg = float(model.predict_proba(row_neg)[:, 1])

    df_notes = pd.DataFrame(
        [
            ["Your note", format_percent(prob_your), "—"],
            ["Positive note", format_percent(prob_pos), f"{(prob_your - prob_pos)*100:+.1f}"],
            ["Neutral note", format_percent(prob_neu), f"{(prob_your - prob_neu)*100:+.1f}"],
            ["High-risk note", format_percent(prob_neg), f"{(prob_your - prob_neg)*100:+.1f}"],
        ],
        columns=["Scenario", "Risk", "Δ vs your note (points)"],
    )

    st.dataframe(df_notes, use_container_width=True, hide_index=True)
    st.caption(
        "The table compares your clinical note to a strongly positive text anchor, "
        "a neutral 'no symptoms' note, and a high-risk symptom note. "
        "Negative Δ means that scenario would have **lower** risk than your current note; "
        "positive Δ means it would be **higher**."
    )

    risk_hits, protect_hits = analyze_note_keywords(note)
    cols_kw = st.columns(2)
    with cols_kw[0]:
        st.markdown("**Risk-raising phrases detected:**")
        if risk_hits:
            for t in risk_hits:
                st.markdown(f"- {t}")
        else:
            st.write("_None of the high-risk keywords were detected._")
    with cols_kw[1]:
        st.markdown("**Protective / reassuring phrases detected:**")
        if protect_hits:
            for t in protect_hits:
                st.markdown(f"- {t}")
        else:
            st.write("_No clearly protective keywords detected._")

    # --------------------------------------------------------
    # Comparison: Traditional-like vs ML
    # --------------------------------------------------------
    st.markdown("### Comparison: Traditional-like vs ML")

    trad_logit = -7.0 + 0.04 * (age - 50) + 0.015 * (sys_bp - 120)
    trad_prob = 1 / (1 + np.exp(-trad_logit))

    st.markdown(
        f"- **Traditional-like score (illustrative, age + SBP only):** {format_percent(trad_prob)}  \n"
        f"- **ML model ({model_name}) risk:** {format_percent(prob)}"
    )
    st.caption(
        "The traditional-like estimate mimics legacy scores that mainly use age and systolic BP. "
        "The ML model additionally incorporates diastolic BP, cholesterol, glucose, BMI, smoking, "
        "family history, and the clinical note."
    )

# ============================================================
# About this App + v1 vs v2 explanation + footer
# ============================================================
with st.expander("About This App"):
    st.markdown(
        """
        ### What this app does

        - Estimates **10-year cardiovascular disease (CVD) risk** for a synthetic patient.  
        - Uses a **GAN-augmented, 5,000-record EHR-like dataset** with:
          age, blood pressure, cholesterol, glucose, BMI, smoking, family history, and clinical notes.  
        - Runs **pre-trained, calibrated machine learning models** (Logistic Regression, XGBoost,
          Gradient Boosting, and a Stacking ensemble).

        ### How the data were built

        1. A base synthetic cohort was generated using clinically realistic distributions  
           (e.g., BP, cholesterol, BMI, smoking, family history).  
        2. A **GAN (Generative Adversarial Network)** was trained on CVD-positive cases.  
           The GAN generates new minority-class records to reach ~30% prevalence while
           preserving realistic feature correlations.  
        3. The GAN-balanced dataset is then used to train the four ML models, followed by
           **probability calibration** (so 20% predicted ≈ 20% observed).

        ### v1 vs v2 (why this app is different)

        - **v1 (`cvd-predictor`)** used a much smaller dataset (~400 records) with limited coverage.  
          Some risk jumps (especially family history) were exaggerated due to small-sample noise.  
        - **v2 (`cvd-predictor2`)** uses a **larger, GAN-balanced synthetic cohort** and
          **fully calibrated models**. Test ROC AUC for the Stacking model is about **0.92**
          with stable cross-validation, so predictions and counterfactuals behave more like
          a clinical-grade risk tool.

        ### Important note

        - This app is for **education, validation, and stakeholder communication** only.  
        - It is **not** trained on real patient-level EHR, and it is **not a medical device**.  
        - Do not use it to make individual treatment decisions.
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
    "It is designed for experimentation, validation, and communication with business and clinical stakeholders, "
    "not for direct patient care."
    "</p>",
    unsafe_allow_html=True,
)
