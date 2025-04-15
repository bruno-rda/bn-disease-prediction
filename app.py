import streamlit as st
from utils.model import load_model, predict, create_pydantic_model

Y_COL = 'Disease'

# Setup session
if 'model' not in st.session_state:
    st.session_state.model = load_model('models/bn_model')
    st.session_state.symptoms = sorted(list(st.session_state.model['model'].nodes)[1:])
    st.session_state.PredictionModel = create_pydantic_model(st.session_state.symptoms)

st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🧠 Disease Prediction App")

# Legend
with st.expander("📘 Legend"):
    st.markdown("""
    Every click on a symptom will toggle between the following states:
    - ❌ Absent (default)
    - ⬜ Uncertain
    - ✅ Present  
    """)

# Utility functions for 3-state checkbox
def toggle_checkbox(key):
    states = ['no', 'none', 'yes']
    current = st.session_state.get(key, 'no')
    next_state = states[(states.index(current) + 1) % len(states)]
    st.session_state[key] = next_state
    # No need to return anything, session_state is updated

def get_icon(state):
    return {
        'yes': '✅',
        'no': '❌',
        'none': '⬜'
    }.get(state, '❌')

# UI layout
st.markdown("### 🔍 Select Symptoms:")
num_cols = 3
cols = st.columns(num_cols)

# Show custom checkboxes
for idx, symptom in enumerate(st.session_state.symptoms):
    key = f"symptom_{symptom}"
    if key not in st.session_state:
        st.session_state[key] = 'no'

    col = cols[idx % num_cols]
    with col:
        btn_col, text_col = st.columns([0.15, 0.85])
        with text_col:
            st.markdown(f"**{symptom.replace('_', ' ').title()}**")
        with btn_col:
            # Use a key that changes with the state to force re-render
            button_key = f"{key}_btn_{st.session_state[key]}"
            if st.button(f"{get_icon(st.session_state[key])}", key=button_key):
                toggle_checkbox(key)
                # Force a rerun to update the UI immediately
                st.rerun()

# Prediction logic
st.markdown("---")
if st.button("🔮 Predict Disease"):
    evidence = {
        symptom: True if st.session_state[f"symptom_{symptom}"] == 'yes'
        else False if st.session_state[f"symptom_{symptom}"] == 'no'
        else None
        for symptom in st.session_state.symptoms
    }

    if any(val is True for val in evidence.values()):
        result = predict(
            st.session_state.model,
            st.session_state.PredictionModel(**evidence),
            Y_COL
        ).head(5).set_index(Y_COL)
        
        st.markdown("### 🧾 Prediction Results:")
        st.bar_chart(result)
        st.dataframe(result)
    else:
        st.warning("⚠️ Select at least one symptom as ✅ Present.")