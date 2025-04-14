import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess import load_data,preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd






# Define the model architecture for 4 classes
def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Define the model architecture f
def create_model_with_dropout(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and save the model
def train_and_save_model(X, y, model_path='models/nnf_model.h5'):
    model = create_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model in H5 format
    model.save(model_path)  # Save model in H5 format

# Train and save the model with dropout
def train_and_save_model_dropout(X, y, model_path='models/dropout_model.h5'):
    model = create_model_with_dropout(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    #os.makedirs("models", exist_ok=True)

    # Save the trained model in H5 format
    model.save(model_path)  # Save model in H5 format

# Train and save the model with dropout
def train_and_save_model_dropout_weights(X, y, model_path='models/dropout_weights_model.h5'):

    # Extract sensitive attribute
    gender_col = X['remainder__sex'].values  # 0 = female, 1 = male (adjust as needed)

    # Compute class weights inversely proportional to gender frequency
    male_weight = 1.0 / np.sum(gender_col == 1)
    female_weight = 1.0 / np.sum(gender_col == 0)

    # Normalize weights so they sum to 1
    total = male_weight * np.sum(gender_col == 1) + female_weight * np.sum(gender_col == 0)
    male_weight /= total
    female_weight /= total

    # Assign weights per sample
    sample_weights = np.where(gender_col == 1, male_weight, female_weight)

    # Train model with sample weights
    model = create_model_with_dropout(X.shape[1])
    model.fit(X.values, y, sample_weight=sample_weights, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save(model_path)



df=load_data()

X=df.drop(columns=['target'])
y=df['target']

num_features=['age','trestbps','chol','thalach','oldpeak']
cat_features=['cp','restecg','slope','ca','thal']
bin_features=['sex','fbs','exang']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_transformed=preprocessor.fit_transform(X_train)
X_test_transformed=preprocessor.transform(X_test)

train_and_save_model(X_train_transformed,y_train)
train_and_save_model_dropout(X_train_transformed,y_train)

#Train and save NNF model with weights
feature_names=preprocessor.get_feature_names_out()
X_train_df=pd.DataFrame(X_train_transformed,columns=feature_names)
train_and_save_model_dropout_weights(X_train_df,y_train)


model1=RandomForestClassifier(n_estimators=67,max_depth=2,min_samples_split=16)
model2=LogisticRegression()

# Saving trained RF model
pipeline = Pipeline(steps=[('pre', preprocessor), ('mdl', model1)])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'models/rf_model.h5')

# Saving trained LR model
pipeline = Pipeline(steps=[('pre', preprocessor), ('mdl', model2)])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'models/lr_model.h5')

# Load the trained model
def load_nnf_model(model_path = 'models/nnf_model.h5'):
    return tf.keras.models.load_model(model_path)

def load_dropout_model(model_path = 'models/dropout_model.h5'):
    return tf.keras.models.load_model(model_path)

def load_dropout_weights_model(model_path = 'models/dropout_weights_model.h5'):
    return tf.keras.models.load_model(model_path)

def load_other_model(model):
    model_path = f'models/{model}_model.h5'
    return joblib.load(model_path)

