import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
df = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\istanbul_ev_kiralik_fiyatlari.csv")


# Define features and target
X = df.drop("Fiyat", axis=1)
y = df["Fiyat"]

# Identify categorical and numerical columns
categorical_c = X.select_dtypes(include="object").columns
numerical_c = X.select_dtypes(include=["float", "int"]).columns

print(f"Categorical Cols: {categorical_c}")
print(f"Numerical Cols: {numerical_c}")

# Define transformations
numerical_trans = Pipeline(steps=[
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numerical_trans, numerical_c),
        ('cat', categorical_transformer, categorical_c)
    ])

# Split data into train and test sets before preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

# Fit and transform the training data, transform the test data
X_train_processed = preprocess.fit_transform(X_train)
X_test_processed = preprocess.transform(X_test)

# Define the model
model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),  # Input shape based on preprocessed data
    Dense(64, activation='relu'),                 # First hidden layer
    Dense(32, activation='relu'),                 # Second hidden layer
    Dense(1, activation='linear')                 # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train_processed, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.25,
    verbose=1
)


y_pred = model.predict(X_test_processed)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Print the results
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

import joblib

# Preprocessing pipeline ve modeli kaydet
joblib.dump(preprocess, "preprocess_pipeline.joblib")
model.save("model.keras")
