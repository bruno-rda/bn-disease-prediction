import streamlit as st
from utils.model import load_model, predict, create_pydantic_model
import plotly.express as px

Y_COL = 'Disease'

# Setup session
if 'model' not in st.session_state:
    st.session_state.model = load_model('models/bn_model')
    st.session_state.symptoms = sorted(list(st.session_state.model['model'].nodes)[1:])
    st.session_state.PredictionModel = create_pydantic_model(st.session_state.symptoms)
    st.session_state.symptoms_btn = {symptom: 0 for symptom in st.session_state.symptoms}

st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🧠 Disease Prediction App")

# Legend
st.markdown("""
Every click on a symptom will toggle between the following states:
- ❌ Absent (default)
- ⬜ Uncertain
- ✅ Present  
---
""")

# Utility functions for 3-state checkbox
def toggle_checkbox(key):
    st.session_state.symptoms_btn[key] = (st.session_state.symptoms_btn[key] + 1) % 3

def get_icon(state):
    return ['❌', '⬜', '✅'][st.session_state.symptoms_btn[state]]

# UI layout
st.markdown(
    "### 🔍 Select Symptoms: <br/>", 
    unsafe_allow_html=True
)

num_cols = 3
cols = st.columns(num_cols)

# Show custom checkboxes
for idx, symptom in enumerate(st.session_state.symptoms):
    col = cols[idx % num_cols]
    with col:
        btn_col, text_col = st.columns([0.15, 0.85])
        with text_col:
            st.markdown(f"**{symptom.replace('_', ' ').title()}**")
        
        with btn_col:
            if st.button(f"{get_icon(symptom)}", key=symptom):
                toggle_checkbox(symptom)
                st.rerun()

# Prediction logic
st.markdown("---")
if st.button("🔮 Predict Disease"):
    evidence = {
        symptom: True if st.session_state.symptoms_btn[symptom] == 2
        else False if st.session_state.symptoms_btn[symptom] == 0
        else None
        for symptom in st.session_state.symptoms
    }

    if any(val is not False for val in evidence.values()):
        result = predict(
            st.session_state.model,
            st.session_state.PredictionModel(**evidence),
            Y_COL
        ).head(5).set_index(Y_COL).rename(columns={'p': 'Probability'})
        
        st.markdown("### 🧾 Prediction Results:")
        st.warning("""
        A prediction below 90% confidence should be taken with a grain of salt.

        This is a demo app. The predictions are not guaranteed to be accurate.
        Please consult a doctor for accurate diagnosis.
        """, icon="⚠️")

        fig = px.bar(
            result,
            y='Probability',
            title='Disease Probability'
        )

        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result)
    else:
        st.warning("⚠️ All symptoms are set to ❌ Absent.")