import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------------------------------
# Page Setup & Theme Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediSymptom AI | Clinical Triage Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for a Modern Clinical Interface
st.markdown("""
<style>
    /* Global Container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .header-sub {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* KPI Summary Cards */
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        min-height: 110px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }

    /* Differential Condition Cards */
    .condition-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    /* Precaution Steps */
    .precaution-item {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.65rem;
        color: #166534;
        font-weight: 500;
        font-size: 0.95rem;
    }

    /* Urgency Badges */
    .triage-mild {
        color: #15803D;
        background: #DCFCE7;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .triage-mod {
        color: #B45309;
        background: #FEF3C7;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .triage-urgent {
        color: #B91C1C;
        background: #FEE2E2;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Ingestion & Caching
# ---------------------------------------------------------
@st.cache_resource
def load_system_assets():
    model = joblib.load("disease_prediction_model.pkl")
    columns = joblib.load("feature_columns.pkl")
    
    desc_df = pd.read_csv("symptom_Description.csv")
    prec_df = pd.read_csv("symptom_precaution.csv")
    sev_df = pd.read_csv("Symptom-severity.csv")

    # Clean text to ensure exact matches
    desc_df['Disease'] = desc_df['Disease'].str.strip()
    prec_df['Disease'] = prec_df['Disease'].str.strip()
    sev_df['Symptom'] = sev_df['Symptom'].str.strip().str.replace(' ', '_')

    # Convert to fast-lookup structures
    desc_dict = dict(zip(desc_df['Disease'], desc_df['Description']))
    prec_dict = prec_df.set_index('Disease').to_dict(orient='index')
    sev_dict = dict(zip(sev_df['Symptom'], sev_df['weight']))

    return model, columns, desc_dict, prec_dict, sev_dict

try:
    model, feature_columns, desc_dict, prec_dict, sev_dict = load_system_assets()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Build label mappings for display
symptom_label_map = {col: col.replace('_', ' ').title() for col in feature_columns}
clean_to_raw_map = {v: k for k, v in symptom_label_map.items()}

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-card">
    <h1 class="header-title">🩺 MediSymptom AI</h1>
    <p class="header-sub">Clinical Decision Support, Differential Diagnosis & Triage System</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Input Panel & Triage Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("📋 Patient Symptoms")
    st.write("Search and select all symptoms the patient is experiencing:")

    selected_display_names = st.multiselect(
        label="Search Symptoms",
        options=sorted(list(symptom_label_map.values())),
        placeholder="e.g., High Fever, Cough, Chills...",
        label_visibility="collapsed"
    )

    selected_symptoms = [clean_to_raw_map[name] for name in selected_display_names]

    st.markdown("---")
    st.markdown("### ℹ️ Operational Guide")
    st.info(
        "• Select **2 to 4 symptoms** for balanced differential evaluation.\n\n"
        "• Top-3 probable diseases are ranked dynamically.\n\n"
        "• Triage level is evaluated using clinical symptom severity weights."
    )

    if st.button("🔄 Reset Inputs", use_container_width=True):
        st.rerun()

# ---------------------------------------------------------
# Main App Body
# ---------------------------------------------------------
if not selected_symptoms:
    # Empty State Cards
    st.info("👈 **Awaiting Clinical Input:** Please select symptoms from the sidebar to generate the diagnostic assessment.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Disease Model</div>
            <div class="metric-value">Random Forest</div>
            <p style="color:#64748B; font-size:0.85rem; margin-top:0.4rem;">Ensemble of 100 decision trees mapped across 41 conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Diagnostic Logic</div>
            <div class="metric-value">Differential</div>
            <p style="color:#64748B; font-size:0.85rem; margin-top:0.4rem;">Multi-class probability ranking to account for symptom co-occurrence.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Triage Metric</div>
            <div class="metric-value">Weighted Severity</div>
            <p style="color:#64748B; font-size:0.85rem; margin-top:0.4rem;">Dynamic urgency scoring calculated from clinical symptom weights.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # 1. Feature Vector Construction
    input_vector = np.zeros(len(feature_columns), dtype=int)
    for symptom in selected_symptoms:
        if symptom in feature_columns:
            idx = feature_columns.index(symptom)
            input_vector[idx] = 1

    input_df = pd.DataFrame([input_vector], columns=feature_columns)

    # 2. Probability Computation & Top-3 Differentials
    probabilities = model.predict_proba(input_df)[0]
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_diseases = [(model.classes_[i], probabilities[i]) for i in top_3_indices]

    primary_disease, primary_confidence = top_3_diseases[0]

    # 3. Clinical Severity & Triage Calculation
    total_severity = sum(sev_dict.get(sym, 1) for sym in selected_symptoms)

    if total_severity <= 13:
        triage_status = "Mild Severity"
        triage_class = "triage-mild"
        triage_icon = "🟢"
        triage_note = "Manageable with monitored self-care and rest."
    elif total_severity <= 22:
        triage_status = "Moderate Severity"
        triage_class = "triage-mod"
        triage_icon = "🟡"
        triage_note = "Medical evaluation by a physician is advised."
    else:
        triage_status = "High Urgency"
        triage_class = "triage-urgent"
        triage_icon = "🔴"
        triage_note = "Prompt professional medical attention advised."

    # 4. Top KPI Metric Summary
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Primary Suspected Prognosis</div>
            <div class="metric-value" style="color:#1E3A8A;">{primary_disease}</div>
            <div style="font-size:0.85rem; color:#64748B; margin-top:0.3rem;">Highest probability match</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value" style="color:#0284C7;">{primary_confidence * 100:.1f}%</div>
            <div style="font-size:0.85rem; color:#64748B; margin-top:0.3rem;">Based on ensemble trees</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Triage Urgency Assessment</div>
            <div style="margin-top: 0.4rem;">
                <span class="{triage_class}">{triage_icon} {triage_status}</span>
            </div>
            <div style="font-size:0.85rem; color:#64748B; margin-top:0.5rem;">Score: <b>{total_severity}</b> ({triage_note})</div>
        </div>
        """, unsafe_allow_html=True)

    # Diagnostic Ambiguity Safety Warning
    if primary_confidence < 0.35:
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "⚠️ **Diagnostic Ambiguity Warning:** The reported symptoms produce a low-confidence classification "
            f"({primary_confidence * 100:.1f}%). Consider selecting additional co-occurring symptoms to improve specificity."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Dual Dedicated Dashboards: Differential vs Precautions
    # -----------------------------------------------------
    dash_col1, dash_col2 = st.columns([1, 1], gap="large")

    # --- LEFT DASHBOARD: Differential Diagnosis & Chart ---
    with dash_col1:
        st.subheader("📊 Differential Diagnosis Analysis")
        st.caption("Top 3 conditions ranked by probability:")

        rank_emojis = ["🥇", "🥈", "🥉"]
        for rank, (dis, prob) in enumerate(top_3_diseases):
            percentage = prob * 100
            st.markdown(f"""
            <div class="condition-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.05rem; font-weight:600; color:#1E293B;">{rank_emojis[rank]} {dis}</span>
                    <span style="font-weight:700; color:#2563EB; font-size:1.05rem;">{percentage:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

        # Interactive Probability Bar Chart
        chart_df = pd.DataFrame({
            'Condition': [d for d, _ in reversed(top_3_diseases)],
            'Probability (%)': [p * 100 for _, p in reversed(top_3_diseases)]
        })

        fig = px.bar(
            chart_df,
            x='Probability (%)',
            y='Condition',
            orientation='h',
            text=[f"{p:.1f}%" for p in chart_df['Probability (%)']],
            color='Probability (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=230,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title=None,
            showlegend=False,
            coloraxis_showscale=False
        )
        fig.update_traces(textposition='inside', textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)

        # Medical Definition Accordion
        description = desc_dict.get(primary_disease, "Clinical overview currently unavailable for this prognosis.")
        with st.expander(f"📖 Clinical Context: {primary_disease}", expanded=True):
            st.write(description)

    # --- RIGHT DASHBOARD: Actionable Precautions & Download ---
    with dash_col2:
        st.subheader("🛡️ Actionable Precautions & Care Plan")
        st.caption(f"Evidence-based guidelines mapped to **{primary_disease}**:")

        precaution_data = prec_dict.get(primary_disease, {})
        precautions = [
            precaution_data.get(f"Precaution_{i}")
            for i in range(1, 5)
            if pd.notna(precaution_data.get(f"Precaution_{i}"))
        ]

        if precautions:
            for idx, item in enumerate(precautions, 1):
                st.markdown(f"""
                <div class="precaution-item">
                    <b>Step {idx}:</b> {item.strip().capitalize()}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("General Precaution: Monitor symptom progression, maintain hydration, and seek physician guidance.")

        st.warning(
            "⚠️ **Clinical Advisory:** MediSymptom AI is an intelligent preliminary triage tool developed for academic "
            "evaluation. It is not an alternative to licensed clinical diagnosis. Seek emergency services for acute symptoms."
        )

        # Downloadable Clinical Summary Report
        st.markdown("### 📄 Patient Diagnostic Report")
        
        precaution_lines = "\n".join([f"  {i}. {p.strip().capitalize()}" for i, p in enumerate(precautions, 1)]) if precautions else "  - Follow general medical advice."
        differential_lines = "\n".join([f"  {idx+1}. {d} ({p*100:.1f}%)" for idx, (d, p) in enumerate(top_3_diseases)])
        
        report_content = f"""==================================================
           MEDISYMPTOM AI - CLINICAL TRIAGE REPORT
==================================================

PATIENT SYMPTOMS:
{', '.join(selected_display_names)}

TRIAGE ASSESSMENT:
- Status: {triage_status}
- Urgency Score: {total_severity}
- Clinical Recommendation: {triage_note}

DIAGNOSTIC FINDINGS:
- Primary Prognosis: {primary_disease}
- Model Confidence: {primary_confidence * 100:.1f}%

DIFFERENTIAL DIAGNOSIS (TOP 3):
{differential_lines}

CLINICAL DESCRIPTION:
{description}

RECOMMENDED ACTIONABLE PRECAUTIONS:
{precaution_lines}

==================================================
DISCLAIMER: This report is generated by an AI decision-support 
system for preliminary triage and academic demonstration only.
==================================================
"""
        st.download_button(
            label="📥 Download Clinical Triage Report (.txt)",
            data=report_content,
            file_name=f"MediSymptom_Report_{primary_disease.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Primary Disease Prediction System • Built with Scikit-Learn, Streamlit, Plotly & Pandas"
    "</div>",
    unsafe_allow_html=True
)