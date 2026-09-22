import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Primary Disease Prediction System",
    page_icon="🩺",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_all_assets():
  # 1. Load ML Model & Features
  model = joblib.load(os.path.join(BASE_DIR, "disease_prediction_model.pkl"))
  features = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

  # 2. Load and normalize Descriptions
  desc_path = os.path.join(BASE_DIR, "symptom_Description.csv")
  desc_df = pd.read_csv(desc_path) if os.path.exists(desc_path) else None
  if desc_df is not None:
    desc_df["clean_disease"] = (
        desc_df["Disease"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("hemorrhoids", "hemmorhoids")
    )

  # 3. Load and normalize Precautions
  prec_path = os.path.join(BASE_DIR, "symptom_precaution.csv")
  prec_df = pd.read_csv(prec_path) if os.path.exists(prec_path) else None
  if prec_df is not None:
    prec_df["clean_disease"] = (
        prec_df["Disease"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("hemorrhoids", "hemmorhoids")
    )

  # 4. Load and map Severity weights
  sev_path = os.path.join(BASE_DIR, "Symptom-severity.csv")
  sev_dict = {}
  if os.path.exists(sev_path):
    sev_df = pd.read_csv(sev_path)
    sev_df = sev_df[sev_df["Symptom"] != "prognosis"]
    sev_df["clean_sym"] = (
        sev_df["Symptom"]
        .astype(str)
        .str.replace("_", " ")
        .str.replace("  ", " ")
        .str.strip()
        .str.lower()
    )
    sev_dict = dict(zip(sev_df["clean_sym"], sev_df["weight"]))

  return model, features, desc_df, prec_df, sev_dict


model, feature_columns, desc_df, prec_df, sev_dict = load_all_assets()

# --- UI Header ---
st.title("🩺 Primary Disease Predictor")
st.markdown(
    "Select your symptoms to predict potential health conditions, compute triage urgency, and review recommended precautions."
)

formatted_options = {
    col: col.replace("_", " ").strip().title() for col in feature_columns
}
selected_display = st.multiselect(
    "Search and select your symptoms:",
    options=sorted(list(formatted_options.values())),
    placeholder="e.g., Itching, Skin Rash, Joint Pain",
)

if st.button("Predict Condition", type="primary"):
  if not selected_display:
    st.warning("Please select at least one symptom.")
  else:
    # Build 0/1 binary input vector
    display_to_col = {v: k for k, v in formatted_options.items()}
    input_data = {col: 0 for col in feature_columns}
    selected_raw = []

    for item in selected_display:
      col_name = display_to_col[item]
      input_data[col_name] = 1
      selected_raw.append(col_name)

    input_df = pd.DataFrame([input_data])

    # Run inference
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    top3_idx = np.argsort(probs)[::-1][:3]

    primary_disease = classes[top3_idx[0]]
    primary_confidence = probs[top3_idx[0]] * 100
    norm_primary_name = (
        primary_disease.strip()
        .lower()
        .replace("hemorrhoids", "hemmorhoids")
        .strip()
    )

    # Calculate Triage Urgency Score
    total_sev_score = 0
    for sym in selected_raw:
      clean_key = sym.replace("_", " ").strip().lower()
      total_sev_score += sev_dict.get(clean_key, 2)  # default weight 2

    # Urgency Level Definition
    if total_sev_score <= 13:
      triage_status = "🟢 Mild (Self-care & monitoring)"
    elif total_sev_score <= 22:
      triage_status = "🟡 Moderate (Consult a physician)"
    else:
      triage_status = "🔴 High Urgency (Immediate clinical care advised)"

    # Low Confidence Banner
    if primary_confidence < 35.0:
      st.warning(
          f"⚠️ **Low Confidence ({primary_confidence:.1f}%)**: The selected symptoms overlap with multiple diseases. Please add more specific symptoms."
      )

    # Display Top Outcome
    st.success(f"### Predicted Condition: **{primary_disease.strip()}**")

    col1, col2 = st.columns(2)
    with col1:
      st.metric("Primary Confidence", f"{primary_confidence:.1f}%")
    with col2:
      st.metric("Symptom Severity", f"{total_sev_score} pts")

    st.caption(f"**Triage Assessment:** {triage_status}")

    # Top 3 Differential Diagnosis
    st.markdown("---")
    st.subheader("Top Differential Diagnosis")
    for idx in top3_idx:
      d_name = classes[idx].strip()
      d_prob = probs[idx] * 100
      st.write(f"**{d_name}** ({d_prob:.1f}%)")
      st.progress(float(d_prob / 100))

    # Condition Description
    if desc_df is not None:
      match_desc = desc_df[desc_df["clean_disease"] == norm_primary_name]
      if not match_desc.empty:
        st.markdown("---")
        st.subheader("About the Condition")
        st.info(match_desc["Description"].values[0])

    # Precautions
    if prec_df is not None:
      match_prec = prec_df[prec_df["clean_disease"] == norm_primary_name]
      if not match_prec.empty:
        st.markdown("---")
        st.subheader("Recommended Precautions")
        precautions = match_prec.iloc[0, 1:5].dropna().tolist()
        for p in precautions:
          st.markdown(f"- {str(p).strip().capitalize()}")