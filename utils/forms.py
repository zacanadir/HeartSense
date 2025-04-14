import streamlit as st
import numpy as np

def user_input_form():
    st.subheader("🧠 Provide Patient Information for Prediction")

    col1, col2 = st.columns(2)

    with st.form("prediction_form"):

        st.markdown("### 🔢 Numerical Features")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=200)
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, value=1.0, step=0.1)
        with col2:
            trestbps = st.number_input("Resting BP (trestbps)", min_value=50, value=120)
            thalach = st.number_input("Max Heart Rate (thalach)", min_value=0, value=150)

        st.divider()

        st.markdown("### 🏷️ Categorical Features")
        col1, col2 = st.columns(2)
        with col1:
            cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, step=1)
            restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2, step=1)
        with col2:
            slope = st.number_input("Slope of ST (slope)", min_value=0, max_value=2, step=1)
            ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, step=1)
            thal = st.number_input("Thalassemia (thal)", min_value=0, max_value=3, step=1)

        st.divider()

        st.markdown("### ⚙️ Binary Features")
        col1, col2 = st.columns(2)
        with col1:
            sex = st.number_input("Sex (1 = male, 0 = female)", min_value=0, max_value=1, step=1)
            fbs = st.number_input("Fasting Blood Sugar > 120 (fbs)", min_value=0, max_value=1, step=1)
            exang = st.number_input("Exercise Induced Angina (exang)", min_value=0, max_value=1, step=1)

        submitted = st.form_submit_button("🚀 Launch Simulation")

    # Return inputs as dictionary if submitted
    if submitted:
        user_input = {
            'age': age, 'trestbps': trestbps, 'chol': chol,
            'thalach': thalach, 'oldpeak': oldpeak,
            'cp': cp, 'restecg': restecg, 'slope': slope, 'ca': ca, 'thal': thal,
            'sex': sex, 'fbs': fbs, 'exang': exang
        }
        st.session_state.age=age
        st.session_state.trestbps=trestbps
        st.session_state.chol=chol
        st.session_state.thalach=thalach
        st.session_state.oldpeak=oldpeak
        st.session_state.cp=cp
        st.session_state.restecg=restecg
        st.session_state.slope=slope
        st.session_state.ca=ca
        st.session_state.thal=thal
        st.session_state.sex=sex
        st.session_state.fbs=fbs
        st.session_state.exang=exang
        st.session_state.form=True
        return user_input
    else:
        return None

def prepare_patient_sample_from_session():
    feature_vector = []

    # Numerical features
    feature_vector.extend([
        st.session_state.age,
        st.session_state.trestbps,
        st.session_state.chol,
        st.session_state.thalach,
        st.session_state.oldpeak
    ])

    # One-hot encoding for categorical features with NO dropped class
    cp = int(st.session_state.cp)
    feature_vector.extend([int(cp == i) for i in range(4)])  # cp_0 to cp_3

    restecg = int(st.session_state.restecg)
    feature_vector.extend([int(restecg == i) for i in range(3)])  # restecg_0 to restecg_2

    slope = int(st.session_state.slope)
    feature_vector.extend([int(slope == i) for i in range(3)])  # slope_0 to slope_2

    ca = int(st.session_state.ca)
    feature_vector.extend([int(ca == i) for i in range(5)])  # ca_0 to ca_4

    thal = int(st.session_state.thal)
    feature_vector.extend([int(thal == i) for i in range(4)])  # thal_0 to thal_3

    # Binary features
    feature_vector.extend([
        st.session_state.sex,
        st.session_state.fbs,
        st.session_state.exang
    ])

    return np.array(feature_vector, dtype=np.float32)

