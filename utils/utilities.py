import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st

def predict_with_mc_dropout(model, sample, n_iterations=100):
    """
    Perform MC Dropout for a single input sample by making multiple predictions
    with dropout layers active at inference time.

    Args:
        model: The Keras model with dropout layers.
        sample: A single input sample (should be 2D: shape (1, n_features)).
        n_iterations: Number of forward passes to simulate dropout.

    Returns:
        predictions: Array of shape (n_iterations, 1, n_classes) with all predictions.
        mean_prediction: Array of shape (1, n_classes), the average prediction.
        std_prediction: Array of shape (1, n_classes), the standard deviation.
    """
    # Get output shape by doing one prediction
    sample_output = model(sample, training=True).numpy()
    predictions = np.zeros((n_iterations, 1, sample_output.shape[1]))  # (iterations, 1, n_classes)

    for i in range(n_iterations):
        preds = model(sample, training=True).numpy()  # (1, n_classes)
        predictions[i] = preds

    mean_prediction = predictions.mean(axis=0)  # (1, n_classes)
    std_prediction = predictions.std(axis=0)    # (1, n_classes)

    return predictions, mean_prediction, std_prediction

import numpy as np
import tensorflow as tf

def predict_with_mc_dropout_and_grads(model, sample, n_iterations=100):
    """
    Perform MC Dropout with gradient tracking for each pass.

    Args:
        model: Keras model with dropout.
        sample: Single input sample, shape (1, n_features), must be float32 tensor or convertible.
        n_iterations: Number of MC samples to generate.

    Returns:
        predictions: Array of shape (n_iterations, 1, n_classes)
        mean_prediction: Array of shape (1, n_classes)
        std_prediction: Array of shape (1, n_classes)
        gradients: Array of shape (n_iterations, n_features) with input gradients
    """
    sample = tf.convert_to_tensor(sample, dtype=tf.float32)
    predictions = []
    gradients = []

    for _ in range(n_iterations):
        with tf.GradientTape() as tape:
            tape.watch(sample)
            preds = model(sample, training=True)  # (1, n_classes)
            # Optional: Choose one class to compute gradient for
            loss = preds[:, 0]  # Gradient of class 0's logit/probability

        grad = tape.gradient(loss, sample)  # shape: (1, n_features)
        predictions.append(preds.numpy())
        gradients.append(grad.numpy()[0])  # strip batch dimension

    predictions = np.array(predictions)       # shape: (n_iterations, 1, n_classes)
    gradients = np.array(gradients)           # shape: (n_iterations, n_features)
    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)

    return predictions, mean_prediction, std_prediction, gradients



def compute_feature_gradients(model, sample, feature_names, class_index=1):
    # Define the numerical (continuous) features you're interested in
    numerical_features = ['trestbps', 'chol', 'thalach', 'oldpeak']

    # Create a binary mask: 1 for numerical features, 0 for others
    mask = np.array([1.0 if name in numerical_features else 0.0 for name in feature_names])

    # Convert sample to tensor
    sample = sample.astype(np.float32)
    sample_tensor = tf.convert_to_tensor(sample.reshape(1, -1), dtype=tf.float32)

    # Watch the input
    with tf.GradientTape() as tape:
        tape.watch(sample_tensor)

        # Feed sample through model
        prediction = model(sample_tensor, training=False)

        # Choose output class (e.g., class 0 for positive diagnosis)
        loss = prediction[:, 0]

    # Get gradients: shape (1, num_features)
    grads = tape.gradient(loss, sample_tensor).numpy().flatten()

    # Zero out gradients for non-numerical features
    grads *= mask
    values = sample.flatten() * mask

    # Filter out zero gradients (non-numerical or zero-valued inputs)
    keep = mask.astype(bool) & (values != 0)

    filtered_features = np.array(feature_names)[keep]
    filtered_grads = grads[keep]
    filtered_vals = values[keep]

    # Wrap into DataFrame
    df_grads = pd.DataFrame({
        'feature': filtered_features,
        'value': filtered_vals,
        'gradient': filtered_grads,
        'abs_gradient': np.abs(filtered_grads)
    }).sort_values(by='gradient', ascending=False)
    df_grads=df_grads[df_grads['gradient']>0]

    return df_grads


def compute_feature_gradients1(model, sample, feature_names):
    # Convert sample to tensor and track it
    sample = tf.convert_to_tensor(sample.reshape(1, -1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(sample)
        prediction = model(sample, training=False)
    grads = tape.gradient(prediction, sample).numpy().flatten()

    # Convert original sample to numpy (to extract the values)
    values = sample.numpy().flatten()

    # Filter out immutable features
    immutable_keywords = ['age', 'sex']
    feature_series = pd.Series(feature_names)
    mask = ~feature_series.apply(lambda name: any(kw in name for kw in immutable_keywords)).to_numpy()

    # Apply the mask to gradients, values, and names
    grads = grads[mask]
    values = values[mask]
    feature_names = np.array(feature_names)[mask]

    # Only keep features where value > 0
    mask_value_positive = values > 0
    grads = grads[mask_value_positive]
    values = values[mask_value_positive]
    feature_names = feature_names[mask_value_positive]

    # Create DataFrame
    df_grads = pd.DataFrame({
        'feature': feature_names,
        'value': values,
        'gradient': grads,
        'abs_gradient': np.abs(grads)
    }).sort_values(by='abs_gradient', ascending=False)

    return df_grads

