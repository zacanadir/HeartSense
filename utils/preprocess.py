import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import  ColumnTransformer
import os



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

