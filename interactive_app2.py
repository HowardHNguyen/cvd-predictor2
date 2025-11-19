import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="CVD 10-Year Risk – ML + GAN (Predictor v2, Hybrid NLP)",
    layout="wide"
)

# Hard-coded evaluation summary from Colab
MODEL_SUMMARY = {
    "Logistic Regression": {"AUC": 0.860, "Brier": 0.104},
    "XGBoost":             {"AUC": 0.913, "Brier": 0.072},
    "Gradient Boosting":   {"AUC": 0.916, "Brier": 0.070},
    "Stacking":            {"AUC": 0.920, "Brier": 0.089},
}

# Anchor clinical notes for comparison in the UI
POSITIVE_NOTE = (
    "active lifestyle, healthy diet, exercises regularly, no symptoms, "
    "never smoked, good sleep, no cardiac complaints"
)

NEGATIVE_NOTE = (
    "chest pain, palpitations, shortness of breath, dizziness on exertion, "
    "fatigue, noncompliant with medications, poor diet, high stress"
)

# Hybrid text-overlay hyperparameters
ALPHA_TEXT = 0.25       # weight of clinical NLP overlay vs base ML risk
TEXT_MAX_EFFECT = 0.30  # max +/- 15 percentage points around prevalence


# ============================================================
# Weighted clinical phrase dictionaries (Hybrid NLP)
# ============================================================
RISK_PHRASES = {
    "chest pain": 3,
    "chest pressure": 3,
    "shortness of breath": 3,
    "dyspnea": 3,
    "orthopnea": 3,
    "paroxysmal nocturnal dyspnea": 3,
    "syncope": 3,
    "fainting": 3,
    "collapse": 3,
    "palpitations": 2,
    "irregular heartbeat": 2,
    "tachycardia": 2,
    "fatigue on exertion": 2,
    "reduced exercise tolerance": 2,
    "leg swelling": 2,
    "ankle swelling": 2,
    "edema": 2,
    "claudication": 2,
    "dizziness": 2,
    "lightheaded": 2,
    "near syncope": 2,
    "prior mi": 2,
    "heart attack history": 2,
    "coronary artery disease": 2,
    "prior stroke": 2,
    "prior tia": 2,
    "atrial fibrillation": 2,
    "a fib": 2,
    "left ventricular hypertrophy": 2,
    "abnormal ekg": 2,
    "ischemic changes": 2,
    "uncontrolled hypertension": 2,
    "poorly controlled blood pressure": 2,
    "poorly controlled bp": 2,
    "hyperlipidemia": 1,
    "elevated cholesterol": 1,
    "diabetes uncontrolled": 2,
    "poor glycemic control": 2,
    "obesity": 1,
    "weight gain": 1,
    "smokes daily": 2,
    "smokes one pack per day": 2,
    "heavy smoker": 2,
    "continues smoking": 2,
    "sedentary lifestyle": 1,
    "physically inactive": 1,
    "poor diet": 1,
    "high stress": 1,
    "noncompliant with medications": 2,
    "stopped medications": 2,
    "poor medication adherence": 2,
    "excessive alcohol": 1,
    "heavy drinking": 1,
}

PROTECT_PHRASES = {
    "active lifestyle": -3,
    "regular exercise": -3,
    "walks daily": -2,
    "goes to gym": -2,
    "swimming regularly": -2,
    "cycling routine": -2,
    "healthy diet": -2,
    "mediterranean diet": -2,
    "low salt diet": -2,
    "low sodium diet": -2,
    "low cholesterol diet": -2,
    "low sugar diet": -2,
    "weight loss progress": -2,
    "losing weight": -2,
    "stable weight": -1,
    "bmi improving": -2,
    "quit smoking": -3,
    "former smoker": -2,
    "reduced smoking": -1,
    "never smoked": -3,
    "no alcohol use": -2,
    "rare alcohol": -1,
    "adequate sleep": -1,
    "good sleep quality": -1,
    "low stress": -2,
    "managing stress well": -1,
    "good support system": -1,
    "no chest pain": -2,
    "no shortness of breath": -2,
    "no dyspnea": -2,
    "no edema": -2,
    "no palpitations": -2,
    "asymptomatic": -2,
    "feels well": -1,
    "normal energy": -1,
    "stable symptoms": -1,
    "bp well controlled": -2,
    "blood pressure well controlled": -2,
    "cholesterol well controlled": -2,
    "diabetes well managed": -2,
    "good glycemic control": -2,
    "good medication adherence": -2,
    "compliant with meds": -2,
    "no missed doses": -1,
    "keeps all appointments": -1,
    "routine follow ups": -1,
    "normal ekg": -2,
    "normal cardiac exam": -2,
    "normal heart sounds": -1,
}


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


def compute_clinical_text_score(note: str) -> int:
    """
    Hybrid clinical NLP score:
    Sum weighted risk phrases (positive) and protective phrases (negative).
    """
    note_l = note.lower()
    score = 0

    for phrase, w in RISK_PHRASES.items():
        if phrase in note_l:
            score += w

    for phrase, w in PROTECT_PHRASES.items():
        if phrase in note_l:
            score += w

    # Clip to avoid extreme values
    return int(np.clip(score, -10, 10))


def analyze_note_keywords(note: str):
    """
    Return lists of which risk and protective phrases were detected.
    """
    note_l = note.lower()
    risk_hits = [p for p in RISK_PHRASES.keys() if p in note_l]
    protect_hits = [p for p in PROTECT_PHRASES.keys() if p in note_l]
    return risk_hits, protect_hits


def scenario_prediction(pipe, base_row: pd.DataFrame, **changes):
    """Clone base_row, apply changes, predict prob (base ML only)."""
    row = base_row.copy()
    for k, v in changes.items():
        row.loc[:, k] = v
    p = float(pipe.predict_proba(row)[:, 1])
    return p


def local_jitter_uncertainty(model, base_row: pd.DataFrame, n_samples: int = 200):
    """
    Estimate local prediction uncertainty by jittering numeric inputs slightly,
    then returning 5th, 50th, 95th percentiles of predicted **base ML** risk.
    (Hybrid text overlay is added later as a constant shift.)
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

PREVALENCE = df["cvd"].mean()  # global prevalence for hybrid text centering

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
    Compute hybrid risk distribution for all rows and each model
    (base ML prob + clinical text overlay) for percentile context.
    """
    X_all = df_full.drop(columns=["cvd"])
    notes = df_full["note"].astype(str)

    # Clinical text scores and overlay risk (same for all models)
    text_scores = notes.apply(compute_clinical_text_score).to_numpy()
    z = np.clip(text_scores, -5, 5)
    p_text = 1 / (1 + np.exp(-z / 2))  # 0..1

    k = TEXT_MAX_EFFECT
    p_text_centered = PREVALENCE + (p_text - 0.5) * k
    p_text_centered = np.clip(p_text_centered, 0.01, 0.99)

    dists = {}

    for name, model in [
        ("Logistic Regression", LR_MODEL),
        ("XGBoost", XGB_MODEL),
        ("Gradient Boosting", GBM_MODEL),
        ("Stacking", STACK_MODEL),
    ]:
        probs_ml = model.predict_proba(X_all)[:, 1]
        probs_hybrid = (1 - ALPHA_TEXT) * probs_ml + ALPHA_TEXT * p_text_centered
        dists[name] = probs_hybrid

    return dists


# Precompute once on import (no caching decorator, depends on model objects)
RISK_DISTS = precompute_risk_distributions(df)

# ============================================================
# Sidebar: dataset + model + patient profile
# ============================================================
with st.sidebar:
    st.markdown("### Dataset status")
    st.success(
        f"Rows: **{len(df):,}**  \n"
        f"Prevalence (CVD=1): **{PREVALENCE*100:.1f}%**"
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
st.title("CVD 10-Year Risk – ML + GAN (Predictor v2, Hybrid NLP)")
st.caption(
    "Demo app using a GAN-augmented 5,000-record synthetic EHR dataset and calibrated ML models, "
    "with a hybrid clinical note overlay. For educational purposes only – not a medical device."
)

left, right = st.columns([1.2, 1.8])

# Build input row and run prediction
row = build_patient_row(age, sys_bp, dia_bp, chol, glu, bmi, smoke, fhx, note)
model = MODEL_MAP[model_name]

with st.spinner("Computing risk and local uncertainty..."):
    # Base ML risk
    prob_ml = float(model.predict_proba(row)[:, 1])

    # Clinical text score and text-based risk overlay
    text_score = compute_clinical_text_score(note)
    z = np.clip(text_score, -5, 5)
    p_text = 1 / (1 + np.exp(-z / 2))  # 0..1

    k = TEXT_MAX_EFFECT
    p_text_centered = PREVALENCE + (p_text - 0.5) * k
    p_text_centered = float(np.clip(p_text_centered, 0.01, 0.99))

    # Blend ML and text overlay → hybrid risk
    prob_final = (1 - ALPHA_TEXT) * prob_ml + ALPHA_TEXT * p_text_centered
    prob_final = float(np.clip(prob_final, 0.01, 0.99))

    # Local jitter-based uncertainty (base ML first)
    low_ml, med_ml, high_ml = local_jitter_uncertainty(model, row, n_samples=200)
    # Apply same hybrid mixing to the jittered band
    low_u = (1 - ALPHA_TEXT) * low_ml + ALPHA_TEXT * p_text_centered
    med_u = (1 - ALPHA_TEXT) * med_ml + ALPHA_TEXT * p_text_centered
    high_u = (1 - ALPHA_TEXT) * high_ml + ALPHA_TEXT * p_text_centered

    # Percentile vs population distribution based on hybrid model
    dist = RISK_DISTS[model_name]
    percentile = get_percentile(prob_final, dist)

# For display, treat prob_final as main risk
prob = prob_final
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
        f"- **Population prevalence (training):** {PREVALENCE*100:.1f}%"
    )

    st.markdown("---")
    st.markdown("### Risk Context in Population")
    st.markdown(
        f"- **Hybrid predicted risk:** {format_percent(prob)}  \n"
        f"- **Population percentile:** ~**{percentile*100:.0f}th** percentile "
        "(higher than this share of the synthetic cohort)."
    )

    st.markdown("---")
    st.markdown("### Model vs Text Contributions")
    st.markdown(
        f"- **Base ML-only risk:** {format_percent(prob_ml)}  \n"
        f"- **Clinical text overlay risk:** {format_percent(p_text_centered)}  \n"
        f"- **Blended (Hybrid) risk:** {format_percent(prob)}  \n"
        f"- **Text score (rule-based):** {text_score:+d}"
    )

# ============================================================
# RIGHT: Risk, uncertainty, counterfactuals, text impact
# ============================================================
with right:
    st.subheader("CVD 10-Year Risk (Hybrid)")

    st.markdown(
        f"<h2 style='color:{risk_color}; margin-bottom:0;'>{format_percent(prob)}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{risk_label}**")

    st.caption(
        "Risk is estimated from a pre-trained, calibrated ML model plus a small, rule-based clinical note overlay. "
        "Inputs include age, BP, cholesterol, glucose, BMI, smoking, family history, and the clinical note."
    )

    # Local uncertainty band
    st.markdown("#### Model Uncertainty (Local, Hybrid)")
    st.markdown(
        f"- **Jittered risk band (5–95%):** {format_percent(low_u)} – {format_percent(high_u)}  \n"
        f"- **Median under jitter:** {format_percent(med_u)}"
    )
    st.caption(
        "Uncertainty is approximated by slightly perturbing numeric inputs (BP, labs, BMI, age) "
        "and recomputing risk 200 times. The band reflects local stability of the hybrid prediction."
    )

    # ------------------------------
    # Counterfactual scenarios (base ML, interpreted as hybrid deltas)
    # ------------------------------
    st.markdown("### What Lowers Risk Most? (Simple Scenarios)")

    scenarios = []

    sbp_lower = max(90, sys_bp - 10)
    p_sbp_ml = scenario_prediction(model, row, sys_bp=sbp_lower)
    p_sbp = (1 - ALPHA_TEXT) * p_sbp_ml + ALPHA_TEXT * p_text_centered
    scenarios.append(("Lower SBP by 10 mmHg", p_sbp))

    dbp_lower = max(50, dia_bp - 5)
    p_dbp_ml = scenario_prediction(model, row, dia_bp=dbp_lower)
    p_dbp = (1 - ALPHA_TEXT) * p_dbp_ml + ALPHA_TEXT * p_text_centered
    scenarios.append(("Lower DBP by 5 mmHg", p_dbp))

    chol_lower = max(120, chol - 20)
    p_chol_ml = scenario_prediction(model, row, cholesterol=chol_lower)
    p_chol = (1 - ALPHA_TEXT) * p_chol_ml + ALPHA_TEXT * p_text_centered
    scenarios.append(("Lower cholesterol by 20 mg/dL", p_chol))

    bmi_lower = max(16.0, bmi - 2.0)
    p_bmi_ml = scenario_prediction(model, row, bmi=bmi_lower)
    p_bmi = (1 - ALPHA_TEXT) * p_bmi_ml + ALPHA_TEXT * p_text_centered
    scenarios.append(("Lower BMI by 2 kg/m²", p_bmi))

    if smoke == "Yes":
        p_qs_ml = scenario_prediction(model, row, smoke=0)
        p_qs = (1 - ALPHA_TEXT) * p_qs_ml + ALPHA_TEXT * p_text_centered
        scenarios.append(("Quit smoking", p_qs))

    if fhx == "Yes":
        p_fh0_ml = scenario_prediction(model, row, family_hx=0)
        p_fh0 = (1 - ALPHA_TEXT) * p_fh0_ml + ALPHA_TEXT * p_text_centered
        scenarios.append(("No family history (counterfactual)", p_fh0))

    if scenarios:
        rows_cf = []
        for desc, p_cf in scenarios:
            abs_change = prob - p_cf
            rows_cf.append({
                "Scenario": desc,
                "New hybrid risk": format_percent(p_cf),
                "Δ risk (points)": f"{abs_change*100:+.1f}",
            })
        cf_df = pd.DataFrame(rows_cf)
        st.dataframe(cf_df, use_container_width=True, hide_index=True)
    else:
        st.write("No counterfactual scenarios defined for this profile.")

    # --------------------------------------------------------
    # Clinical Note impact and keywords (hybrid)
    # --------------------------------------------------------
    st.markdown("### Clinical Note Impact (Hybrid)")

    def hybrid_risk_for_note(note_text: str) -> float:
        """Compute full hybrid risk for a given note text."""
        r = row.copy()
        r.loc[:, "note"] = note_text
        p_ml = float(model.predict_proba(r)[:, 1])
        t_score = compute_clinical_text_score(note_text)
        z_local = np.clip(t_score, -5, 5)
        p_text_local = 1 / (1 + np.exp(-z_local / 2))
        p_text_centered_local = PREVALENCE + (p_text_local - 0.5) * TEXT_MAX_EFFECT
        p_text_centered_local = float(np.clip(p_text_centered_local, 0.01, 0.99))
        p_hybrid = (1 - ALPHA_TEXT) * p_ml + ALPHA_TEXT * p_text_centered_local
        return float(np.clip(p_hybrid, 0.01, 0.99))

    prob_your = prob
    prob_pos = hybrid_risk_for_note(POSITIVE_NOTE)
    prob_neu = hybrid_risk_for_note("no symptoms reported")
    prob_neg = hybrid_risk_for_note(NEGATIVE_NOTE)

    df_notes = pd.DataFrame(
        [
            ["Your note", format_percent(prob_your), "—"],
            ["Positive note", format_percent(prob_pos), f"{(prob_your - prob_pos)*100:+.1f}"],
            ["Neutral note", format_percent(prob_neu), f"{(prob_your - prob_neu)*100:+.1f}"],
            ["High-risk note", format_percent(prob_neg), f"{(prob_your - prob_neg)*100:+.1f}"],
        ],
        columns=["Scenario", "Hybrid risk", "Δ vs your note (points)"],
    )

    st.dataframe(df_notes, use_container_width=True, hide_index=True)
    st.caption(
        "The table compares your clinical note to a strongly positive anchor, "
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
    # Comparison: Traditional-like vs Hybrid ML
    # --------------------------------------------------------
    st.markdown("### Comparison: Traditional-like vs Hybrid ML")

    trad_logit = -7.0 + 0.04 * (age - 50) + 0.015 * (sys_bp - 120)
    trad_prob = 1 / (1 + np.exp(-trad_logit))

    st.markdown(
        f"- **Traditional-like score (illustrative, age + SBP only):** {format_percent(trad_prob)}  \n"
        f"- **Hybrid ML risk ({model_name} + clinical NLP):** {format_percent(prob)}"
    )
    st.caption(
        "The traditional-like estimate mimics legacy scores that mainly use age and systolic BP. "
        "The hybrid ML model additionally incorporates diastolic BP, cholesterol, glucose, BMI, smoking, "
        "family history, and the clinical note."
    )

# ============================================================
# About this App + v1 vs v2 explanation + footer
# ============================================================
with st.expander("About This App"):
    st.markdown(
        f"""
        ### What this app does

        - Estimates **10-year cardiovascular disease (CVD) risk** for a synthetic patient.  
        - Uses a **GAN-augmented, 5,000-record EHR-like dataset** with:
          age, blood pressure, cholesterol, glucose, BMI, smoking, family history, and clinical notes.  
        - Runs **pre-trained, calibrated machine learning models** (Logistic Regression, XGBoost,
          Gradient Boosting, and a Stacking ensemble).  
        - Adds a small, interpretable **hybrid clinical NLP overlay** based on rule-based keyword scoring.

        ### How the data were built

        1. A base synthetic cohort was generated using clinically realistic distributions  
           (e.g., BP, cholesterol, BMI, smoking, family history).  
        2. A **GAN (Generative Adversarial Network)** was trained on CVD-positive cases.  
           The GAN generates new minority-class records to reach ~{PREVALENCE*100:.1f}% prevalence while
           preserving realistic feature correlations.  
        3. The GAN-balanced dataset is then used to train the four ML models, followed by
           **probability calibration** (so 20% predicted ≈ 20% observed).

        ### Hybrid clinical NLP overlay

        - The clinical note is interpreted in **two ways**:
          - Automatically via the TF-IDF features inside the ML model.  
          - Via a **rule-based phrase dictionary** that scores red-flag symptoms (e.g., 'chest pain')
            and reassuring phrases (e.g., 'no chest pain', 'active lifestyle').  
        - The rule-based score is converted into a text risk and blended with the ML risk with weight **{ALPHA_TEXT:.2f}**,
          capped so it can adjust the probability by at most **±{TEXT_MAX_EFFECT*50:.0f} percentage points**
          around the population baseline.

        ### v1 vs v2 (why this app is different)

        - **v1 (`cvd-predictor`)** used a much smaller dataset (~400 records) with limited coverage.  
          Some risk jumps (especially family history) were exaggerated due to small-sample noise.  
        - **v2 (`cvd-predictor2`, this app)** uses a **larger, GAN-balanced synthetic cohort** and
          **fully calibrated models** with a **hybrid, interpretable clinical NLP overlay**.
          Test ROC AUC for the Stacking model is about **0.92** with stable cross-validation, so
          predictions and counterfactuals behave more like a clinical-grade risk tool.

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
    "This tool is built on synthetic, GAN-augmented data and calibrated ML models, "
    "with a hybrid clinical note overlay. It is designed for experimentation, validation, and "
    "communication with business and clinical stakeholders, not for direct patient care."
    "</p>",
    unsafe_allow_html=True,
)
