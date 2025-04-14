import streamlit as st
from utils.preprocess import load_data, preprocessor
from model import load_nnf_model,load_other_model,load_dropout_model,load_dropout_weights_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input
from utils.forms import user_input_form,prepare_patient_sample_from_session
from utils.utilities import predict_with_mc_dropout,compute_feature_gradients,predict_with_mc_dropout_and_grads
from utils.figures import plot_kde_mc_dropout,plot_accuracy_analysis,plot_bias_analysis
import pandas as pd
import plotly.express as px
import shap
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

st.set_page_config(
    page_title="HeartSense: Uncertainty-Aware Heart Disease Prediction & Explanation",
    page_icon="🫀",
    layout="wide"
)


st.title("🫀 HeartSense: Uncertainty-Aware Heart Disease Prediction & Explanation")
# 1. Introduction Section
st.header("🫀 App Overview: Predicting and Understanding Heart Disease Risk")
st.write(f"""This interactive app empowers users to predict the likelihood of heart disease and understand the key drivers behind the risk using a Monte Carlo-based neural network model.

We start by evaluating and comparing three predictive algorithms — Logistic Regression (LR), Random Forest (RF), and a custom Neural Network with Dropout (NNF) — using ROC and Precision-Recall curves for a dynamic view of performance. We also assess gender fairness by examining these metrics separately for men and women, and mitigate bias using class reweighting in the NNF model.

The heart of the app lies in its uncertainty-aware prediction system. Users can input patient data to receive a probabilistic prediction of heart disease risk, visualized through a distribution (KDE) generated from multiple forward passes via Monte Carlo Dropout.

Finally, we identify consequential features — the patient variables that most influence prediction — using gradient analysis. For controllable continuous features, we compute and visualize 95% confidence intervals for their impact, helping users target the most effective areas for prevention or intervention.""")

st.subheader("🧠 Why It Matters")
st.info("Rather than treating model predictions as fixed, HeartSense embraces uncertainty and interpretability to offer a more nuanced, actionable view of health risk — empowering users and practitioners to focus on what matters most for each individual case.")
df=load_data()

X=df.drop(columns=['target'])
y=df['target']

num_features=['age','trestbps','chol','thalach','oldpeak']
cat_features=['cp','restecg','slope','ca','thal']
bin_features=['sex','fbs','exang']
features=num_features+cat_features+bin_features


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train_transformed=preprocessor.fit_transform(X_train)
X_test_transformed=preprocessor.transform(X_test)
f_names_encoded=preprocessor.get_feature_names_out()
f_names = [name.split('__')[-1] for name in f_names_encoded]  # Strip 'scaler__' or 'remainder__'


model_nnf=load_nnf_model()
model1=load_other_model('rf')
model2=load_other_model('lr')

#model_nnf_dropout_balanced=load_dropout_weights_model()
# --- Predict with NNF ---
y_pred_nnf = model_nnf.predict(X_test_transformed)
y_class_nnf = (y_pred_nnf > 0.5).astype(int).flatten()


#y_pred_nnf = model_nnf_dropout_balanced.predict(X_test_transformed)
#y_class_nnf = (y_pred_nnf > 0.5).astype(int).flatten()


# Traditional models (RF, LR)
models = {'RF': model1, 'LR': model2}

st.header("Model Comparison")
st.write("Evaluate and compare Logistic Regression, Random Forest, and a custom Neural Network using ROC and Precision-Recall curves.")
plot_accuracy_analysis(models,X_test,y_test,y_pred_nnf)

st.header("Fairness Analysis")
st.write("Analyze model performance separately for men and women, and apply reweighting techniques to mitigate bias.")
plot_bias_analysis(X_test,y_test,models,y_pred_nnf)

# TODO: Add weights
st.info(
    "**⚠️ Gender Bias Detected:** The analysis indicates a performance disparity across gender groups, "
    "with the model exhibiting bias against male patients. To address this, we will incorporate group-aware "
    "sample weights during neural network training. This updated model will be used in all downstream predictions "
    "and explanations to improve fairness and reduce bias-related harm."
)

model_nnf_dropout_balanced=load_dropout_weights_model()

st.header("Uncertainty-Aware Prediction")
st.write("Make personalized predictions using Monte Carlo Dropout to visualize risk as a probability distribution, not just a static number.")
user_input = user_input_form()

if user_input:
    sample = prepare_patient_sample_from_session().reshape(1, -1)
    with st.spinner("🔄 Running Monte Carlo Simulation... Please wait"):
        preds, mean, std = predict_with_mc_dropout(model_nnf_dropout_balanced, sample,n_iterations=10_000
        )
    st.success("✅ Simulation complete!")

    #preds, mean, std = predict_with_mc_dropout(model_nnf_dropout_balanced, sample)

    plot_kde_mc_dropout(preds)

st.header("Consequential Feature Identification")
st.write("Use gradient-based analysis to uncover which patient features most influence disease risk — with 95% confidence intervals for controllable, continuous variables.")

st.info(
    "⚠️ Only continuous and controllable features "
    "(`trestbps`, `chol`, `thalach`, `oldpeak`) are considered here. "
    "This is because the method evaluates **gradients**, which estimate how small changes in input affect the prediction. "
    "Categorical or discrete variables can cause unpredictable results in this context. "
    "Other methods, like mixed-integer optimization, may be better suited for exploring those variables."
)

if st.button("Show Consequential Features (Gradient-Based)"):
    if st.session_state.get("form", False):
        sample = prepare_patient_sample_from_session().reshape(1, -1)
        #gradient_df = compute_feature_gradients(predict_with_mc_dropout_and_grads, sample, feature_names=f_names)
        #_,_,_,gradient_grid=predict_with_mc_dropout_and_grads(model_nnf_dropout_balanced, sample)
        with st.spinner("🔄 Running Monte Carlo Simulation... Please wait"):
            predictions, mean_prediction, std_prediction, gradient_grid = predict_with_mc_dropout_and_grads(
                model_nnf_dropout_balanced, sample, n_iterations=10_000
            )
        st.success("✅ Simulation complete!")

        # Compute CIs for controllable continuous features
        controllable = ['trestbps', 'chol', 'thalach', 'oldpeak']
        controllable_indices = [f_names.index(f) for f in controllable]
        grads_controllable = gradient_grid[:, controllable_indices]

        # Compute stats
        mean_grad = grads_controllable.mean(axis=0)
        lower = np.percentile(grads_controllable, 2.5, axis=0)
        upper = np.percentile(grads_controllable, 97.5, axis=0)

        # Build DataFrame for plotting
        ci_df = pd.DataFrame({
            'Feature': controllable,
            'Mean Gradient': mean_grad,
            'Lower CI': lower,
            'Upper CI': upper
        })

        # Plot with error bars
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ci_df['Feature'],
            y=ci_df['Mean Gradient'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=ci_df['Upper CI'] - ci_df['Mean Gradient'],
                arrayminus=ci_df['Mean Gradient'] - ci_df['Lower CI'],
                thickness=2,
                width=6
            ),
            mode='markers+lines',
            marker=dict(color='crimson', size=10),
            line=dict(color='crimson', width=2),
            name="Gradient ± 95% CI"
        ))

        fig.update_layout(
            title="📈 95% Confidence Interval for Feature Gradients (Controllable Features)",
            xaxis_title="Feature",
            yaxis_title="∂P(disease)/∂feature",
            template='plotly_white',
            height=450,
            margin=dict(t=60, l=20, r=20, b=60)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Each point represents the **average effect** of a feature on the predicted probability, "
            "with error bars showing the **95% confidence interval** from MC simulations. "
            "Large intervals suggest uncertain or unstable influence."
        )
    else:
        st.warning("Please submit the form first.")
