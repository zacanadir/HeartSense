import pandas as pd
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, Model, Input
import tensorflow as tf



num_features=['age','trestbps','chol','thalach','oldpeak']
cat_features=['cp','restecg','slope','ca','thal']
bin_features=['sex','fbs','exang']

def load_data():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(parent_dir, 'data', 'heart.csv')
    return pd.read_csv(data_path)

preprocessor=ColumnTransformer(transformers=[('scaler',StandardScaler(),num_features),
                                             ('encoder',OneHotEncoder(handle_unknown='ignore',drop=None),cat_features)
                                             ],remainder='passthrough'
                               )

