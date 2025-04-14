import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import streamlit as st
from sklearn.metrics import roc_auc_score,average_precision_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_accuracy_analysis(models,X_test,y_test,y_pred_nnf):
    # Create 1-row, 2-column subplot
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve"))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc:.4f})"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f"{name} (AP={avg_precision:.4f})"),
                      row=1, col=2)

    # NNF (neural net with dropout)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nnf)
    roc_auc_nn = roc_auc_score(y_test, y_pred_nnf)

    precision_nn, recall_nn, _ = precision_recall_curve(y_test, y_pred_nnf)
    avg_precision_nn = average_precision_score(y_test, y_pred_nnf)

    fig.add_trace(go.Scatter(x=fpr_nn, y=tpr_nn, mode='lines', name=f"NNF (AUC={roc_auc_nn:.4f})"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=recall_nn, y=precision_nn, mode='lines', name=f"NNF (AP={avg_precision_nn:.4f})"),
                  row=1, col=2)

    # Layout tweaks
    fig.update_layout(
        template="plotly_white",
        width=1000,
        height=500,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )

    # Set axis labels for both subplots
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_bias_analysis(X_test,y_test,models,y_pred_nnf):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    group_labels = {0: 'Female', 1: 'Male'}
    X_test_copy = X_test.copy()
    X_test_copy['y_true'] = y_test.values

    # Set up 2 rows (ROC, PR) × 3 columns (one per model)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "ROC - RF", "ROC - LR", "ROC - NNF",
            "PR - RF", "PR - LR", "PR - NNF"
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.2
    )

    # Helper to assign column from model name
    model_col_map = {"RF": 1, "LR": 2, "NNF": 3}

    # Evaluate scikit-learn models
    for name, model in models.items():
        col = model_col_map[name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        X_test_copy['y_pred'] = y_pred_proba

        for group in X_test_copy['sex'].unique():
            group_name = group_labels.get(group, group)
            subset = X_test_copy[X_test_copy['sex'] == group]

            fpr, tpr, _ = roc_curve(subset['y_true'], subset['y_pred'])
            precision, recall, _ = precision_recall_curve(subset['y_true'], subset['y_pred'])

            roc_auc = auc(fpr, tpr)
            pr_auc = auc(recall, precision)

            # ROC
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name} - {group_name} (AUC={roc_auc:.2f})",
                showlegend=(col == 1)  # show legend only once
            ), row=1, col=col)

            # PR
            fig.add_trace(go.Scatter(
                x=recall, y=precision, mode='lines',
                name=f"{name} - {group_name} (AUC={pr_auc:.2f})",
                showlegend=False
            ), row=2, col=col)

    # Neural Net (already predicted externally)
    X_test_copy['y_pred'] = y_pred_nnf
    name = "NNF"
    col = model_col_map[name]

    for group in X_test_copy['sex'].unique():
        group_name = group_labels.get(group, group)
        subset = X_test_copy[X_test_copy['sex'] == group]

        fpr, tpr, _ = roc_curve(subset['y_true'], subset['y_pred'])
        precision, recall, _ = precision_recall_curve(subset['y_true'], subset['y_pred'])

        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        # ROC
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"NNF - {group_name} (AUC={roc_auc:.2f})",
            showlegend=False
        ), row=1, col=col)

        # PR
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode='lines',
            name=f"NNF - {group_name} (AUC={pr_auc:.2f})",
            showlegend=False
        ), row=2, col=col)

    # Layout and styling
    fig.update_layout(
        height=700,
        width=1000,
        title_text="Fairness: ROC and PR Curves by Sex and Model",
        template="plotly_white"
    )

    # Axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=i)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=i)
        fig.update_xaxes(title_text="Recall", row=2, col=i)
        fig.update_yaxes(title_text="Precision", row=2, col=i)

    # Show in Streamlit
    st.plotly_chart(fig)


def plot_kde_mc_dropout(preds):
    """
    Plot KDE of MC Dropout predictions for a single sample.
    Also prints mean, std, and notes on multimodality.
    """
    # Drop singleton dimensions and get 1D array of predicted probabilities for class 1
    sample_preds = np.squeeze(preds)  # (n_iterations,) if binary, or (n_iterations, n_classes)

    # Handle multi-class case: optionally plot class 1 probability
    if sample_preds.ndim == 2 and sample_preds.shape[1] > 1:
        sample_preds = sample_preds[:, 1]  # use class 1 probability

    # Compute KDE
    kde = gaussian_kde(sample_preds)
    x_vals = np.linspace(0, 1, 200)
    y_vals = kde(x_vals)

    # Compute stats
    mean_val = np.round(np.mean(sample_preds), 3)
    std_val = np.round(np.std(sample_preds), 3)
    Q1 = np.percentile(sample_preds, 25)
    Q3 = np.percentile(sample_preds, 75)
    IQR = Q3 - Q1

    # Detect multiple modes
    peaks, _ = find_peaks(y_vals, prominence=0.01)
    n_modes = len(peaks)

    #minima, _ = find_peaks(-y_vals, prominence=0.01)
    #n_modes = len(minima) + 1

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        fill='tozeroy',
        name='KDE',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        title="MC Dropout Prediction Distribution (KDE)",
        xaxis_title="Predicted Probability",
        yaxis_title="Density",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary report
    st.markdown(f"""
    ### Prediction Summary
    - **Mean prediction:** {mean_val}
    - **Uncertainty (std):** {std_val}
    - **Interquartile Range (IQR):** {IQR:.4f}
    """)

    if n_modes > 1 or IQR > 0.3:
        st.warning(
            f"The predicted probability distribution shows **{n_modes} distinct mode(s)** "
            f"and an interquartile range (IQR) of **{IQR:.2f}**, indicating notable uncertainty. "
            f"This suggests the model considers multiple plausible outcomes for this input."
        )
    else:
        st.success(
            f"The distribution is **unimodal** with a tight IQR of **{IQR:.2f}**, "
            f"indicating higher confidence in the predicted outcome."
        )
