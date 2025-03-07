# ==============================================================================
# 🔍 Anomaly Detection Pipeline - Data Loading & Inspection
# ==============================================================================
# This script loads and verifies preprocessed sensor data for each machine:
# - Cylinder Bottom Saw
# - CNC Lathe (Piston Rod)
# - CNC Milling Machine (Cylinder Bottom)
# ==============================================================================

import os  # File paths
import sys  # System-specific parameters    
import logging # to log messages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib   # to save and load models
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_curve, auc)
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# =====================================================================
# ✅ Load and Inspect Processed Sensor Data
# =====================================================================

def load_sensor_data(file_path):
    """Loads a processed CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# File Paths
saw_csv_path = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\processed\CYLINDER_BOTTOM_SAW\saw_internal_signal.csv"
lathe_csv_path = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\processed\PISTON_ROD_CNC_LATHE\cnc_lathe_internal_signal.csv"

# Load Data
saw_df = load_sensor_data(saw_csv_path)
lathe_df = load_sensor_data(lathe_csv_path)

# Debugging Check
print(saw_df.head())
print(lathe_df.head())
print(saw_df.info())
print(lathe_df.info())
print(saw_df.describe())
print(lathe_df.describe())
print(saw_df.columns)
print(lathe_df.columns)
print(saw_df.shape)
print(lathe_df.shape)     

#-------------------------------------------
# Convert timestamp column to datetime
saw_df['timestamp'] = pd.to_datetime(saw_df['timestamp'])
lathe_df['timestamp'] = pd.to_datetime(lathe_df['timestamp'])

# =====================================================================
# 📊 Time-Series Visualization of Sensor Data
# =====================================================================
def plot_time_series(df, default_sensor_column, title):
    """Plots time-series data of a given sensor reading."""
    if "timestamp" not in df.columns:
        raise KeyError("❌ 'timestamp' column is missing from DataFrame! Please check your dataset.")

    sensor_column = default_sensor_column if default_sensor_column in df.columns else df.columns[1]

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df[sensor_column], label="Sensor Reading", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel(sensor_column)
    plt.title(title)
    plt.legend()
    plt.show()

# Call the function with updated logic
plot_time_series(saw_df, "SpindleRPM", "Cylinder Bottom Saw")
plot_time_series(lathe_df, "SpindleRPM", "CNC Lathe (Piston Rod)")

#-------------------------------------------
# Overlay anomaly markers on the graphs to highlight unusual behavior based on the anomaly labels in your dataset.

# Ensure data is sorted by timestamp
saw_df = saw_df.sort_values(by="timestamp")
lathe_df = lathe_df.sort_values(by="timestamp")

# Identify anomaly points
saw_anomalies = saw_df[saw_df["anomaly"] == 1]
lathe_anomalies = lathe_df[lathe_df["anomaly"] == 1]

# Define sensor columns
saw_sensor_column = "CPU_cooler_temp"
lathe_sensor_column = "SpindleRPM"

# Plot Cylinder Bottom Saw with Anomalies
plt.figure(figsize=(12, 6))
plt.plot(saw_df["timestamp"], saw_df[saw_sensor_column], label="Sensor Reading", alpha=0.7)
plt.scatter(saw_anomalies["timestamp"], saw_anomalies[saw_sensor_column], color="red", label="Anomalies", zorder=3)
plt.xlabel("Time")
plt.ylabel(saw_sensor_column)
plt.title("Cylinder Bottom Saw - Sensor Data with Anomalies")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Plot CNC Lathe with Anomalies
plt.figure(figsize=(12, 6))
plt.plot(lathe_df["timestamp"], lathe_df[lathe_sensor_column], label="Sensor Reading", alpha=0.7)
plt.scatter(lathe_anomalies["timestamp"], lathe_anomalies[lathe_sensor_column], color="red", label="Anomalies", zorder=3)
plt.xlabel("Time")
plt.ylabel(lathe_sensor_column)
plt.title("CNC Lathe (Piston Rod) - Sensor Data with Anomalies")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# =====================================================================
# 🔥 Pairplot (Scatter Matrix) 
# =====================================================================
# Select a subset of features for pairplot to avoid excessive computation
selected_features_saw = ["CPU_temp", "CutTime", "PData.CutEnergy", "blade_position", "Vib01.Peak", "Vib02.Peak", "anomaly"]
selected_features_lathe = ["SpindleRPM", "actSpeed1", "aaTorque1", "aaLoad1", "aaPower1", "TimeSinceStartup", "anomaly"]

# Generate Pairplot for Cylinder Bottom Saw
sns.pairplot(saw_df[selected_features_saw], hue="anomaly", diag_kind="kde")
plt.suptitle("Cylinder Bottom Saw - Pairplot (Scatter Matrix)", y=1.02)
plt.show()

# Generate Pairplot for CNC Lathe
sns.pairplot(lathe_df[selected_features_lathe], hue="anomaly", diag_kind="kde")
plt.suptitle("CNC Lathe - Pairplot (Scatter Matrix)", y=1.02)
plt.show()

# ==============================================================================
# ✅ Function: Handle Missing Values
# ==============================================================================
def handle_missing_values(df):
    """Handles missing values by dropping columns with >50% missing data and filling missing values."""
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(method="ffill")
        else:
            df[column] = df[column].fillna(df[column].median())

    logging.info("✅ Missing values handled successfully.")
    return df

# Handle missing values
cylinder_bottom_saw_df = handle_missing_values(saw_df)
piston_rod_cnc_lathe_df = handle_missing_values(lathe_df)

# ==============================================================================
# ✅ Function: Remove Highly Correlated Features
# ==============================================================================
def remove_correlated_features(df, correlation_threshold=0.9):
    """Removes highly correlated features (> threshold)."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    
    df = df.drop(columns=to_drop)
    logging.info(f"✅ Removed highly correlated features: {to_drop}")
    return df

# Remove highly correlated features
cylinder_bottom_saw_df = remove_correlated_features(cylinder_bottom_saw_df)
piston_rod_cnc_lathe_df = remove_correlated_features(piston_rod_cnc_lathe_df)

# ==============================================================================
# ✅ Scale Features
# ==============================================================================
from sklearn.preprocessing import StandardScaler

# Drop the timestamp column (it's not needed for scaling)
saw_df_numeric = cylinder_bottom_saw_df.drop(columns=["timestamp"])
lathe_df_numeric = piston_rod_cnc_lathe_df.drop(columns=["timestamp"])

# Apply Standard Scaling
scaler = StandardScaler()
saw_df_scaled = pd.DataFrame(scaler.fit_transform(saw_df_numeric), columns=saw_df_numeric.columns)
lathe_df_scaled = pd.DataFrame(scaler.fit_transform(lathe_df_numeric), columns=lathe_df_numeric.columns)

# ==============================================================================
# ✅ Remove Low-Variance Features
# ==============================================================================
from sklearn.feature_selection import VarianceThreshold

def remove_low_variance_features(df, threshold=0.01):
    """Removes features with variance lower than the given threshold."""
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    selected_features = df.columns[selector.get_support()]
    
    logging.info(f"✅ Features retained after variance thresholding: {list(selected_features)}")
    return df[selected_features]

# Apply variance thresholding
saw_df_filtered = remove_low_variance_features(saw_df_scaled)
lathe_df_filtered = remove_low_variance_features(lathe_df_scaled)

#==============================================================================
 # ✅  Scale Features

"""     
we have two datasets:
Cylinder Bottom Saw → (332,847 samples, 46 features)
CNC Lathe (Piston Rod) → (504,826 samples, 92 features)

since Autoencoders and Machine Learning Models Work Best with Normalized Data

Many ML algorithms (especially deep learning models) assume that all features are on a similar scale.
Autoencoders use neural networks, which are sensitive to the scale of input features.
Prevents Dominance of Features with Large Ranges

Some features (e.g., CutTime in milliseconds) have much larger ranges than others (e.g., Vib01.Peak).
Without scaling, features with large values dominate the training process, making the model ineffective.
Improves Anomaly Detection Performance

Since anomalies are outliers in feature space, bringing all features to the same scale helps the model learn normal patterns better.

"""
from sklearn.preprocessing import StandardScaler

# ✅ Drop the timestamp column (it's not needed for scaling)
"""Before applying PCA, the anomaly column is explicitly excluded from the feature data
   If the anomaly column was included in the PCA transformation, 
   it would have been treated as a feature and transformed into continuous values, which is not what we want.
"""
saw_df_numeric = saw_df.drop(columns=["timestamp"])
lathe_df_numeric = lathe_df.drop(columns=["timestamp"])

# ✅ Apply Standard Scaling
scaler = StandardScaler()
saw_df_scaled = pd.DataFrame(scaler.fit_transform(saw_df_numeric), columns=saw_df_numeric.columns)
lathe_df_scaled = pd.DataFrame(scaler.fit_transform(lathe_df_numeric), columns=lathe_df_numeric.columns)

# ✅ Check the shapes after scaling
print("Cylinder Bottom Saw (scaled):", saw_df_scaled.shape)
print("CNC Lathe (scaled):", lathe_df_scaled.shape)


#print normal and anomaly data to check they are not there
print(saw_df_scaled[saw_df_scaled["anomaly"] == 0].head())
print(saw_df_scaled[saw_df_scaled["anomaly"] == 1].head())

#---------------------------------------------------------
# ✅  Remove Low-Variance Features

"""remove_low_variance_features(df)
The feature variance thresholding should be applied AFTER scaling because:

Raw features might have different scales → Features with large magnitude values (e.g., CutTime, SpindleRPM) will naturally have higher variance, 
while small-magnitude features (e.g., CosPhi, Vibration Skewness) may have low variance, even if they are informative.
Standard scaling normalizes variance across features → Ensures that low variance truly means low variance and isn't just due to different numerical scales."""
#---------------------------------------------------------
from sklearn.feature_selection import VarianceThreshold

def remove_low_variance_features(df, threshold=0.01):
    """Removes features with variance lower than the given threshold."""
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    selected_features = df.columns[selector.get_support()]
    
    logging.info(f"✅ Features retained after variance thresholding: {list(selected_features)}")
    return df[selected_features]

# ✅ Apply variance thresholding
saw_df_filtered = remove_low_variance_features(saw_df_scaled)
lathe_df_filtered = remove_low_variance_features(lathe_df_scaled)

# ✅ Check new shapes after removing low-variance features
print("Cylinder Bottom Saw (filtered):", saw_df_filtered.shape)
print("CNC Lathe (filtered):", lathe_df_filtered.shape)



#==============================================================================
# ✅ Apply PCA for Dimensionality Reduction

"""
Currently, your datasets have:

Cylinder Bottom Saw → 332,847 samples, 46 features
CNC Lathe (Piston Rod) → 504,826 samples, 92 features
Training an autoencoder with all features can be computationally expensive and may include irrelevant or 
highly correlated features that can reduce anomaly detection performance.

 So let us apply PCA (Principal Component Analysis) for feature reduction,
 PCA will help reduce the dimensionality of the data while retaining most of the variance,
 which is especially useful for improving computational efficiency and simplifying your models.
 
 PCA does not affect the labels (normal vs. anomaly), so their distribution remains unchanged.

The feature space is transformed into principal components, which can improve the performance and interpretability of your models.
 """


# ==============================================================================
# ✅ Apply PCA for Dimensionality Reduction
# ==============================================================================
from sklearn.decomposition import PCA

def apply_pca(df, n_components=0.95):
    """
    Applies PCA to reduce the dimensionality of the dataset.
    
    Parameters:
        df (pd.DataFrame): The input dataset (scaled and filtered).
        n_components (float or int): Number of components to retain. If float, it represents the explained variance ratio.
    
    Returns:
        pd.DataFrame: The transformed dataset with reduced dimensions.
        PCA: The fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    
    # Convert back to DataFrame for easier handling
    df_pca = pd.DataFrame(df_pca, columns=[f"PC{i+1}" for i in range(df_pca.shape[1])])
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"✅ Explained variance ratio: {explained_variance:.2f}")
    print(f"✅ Number of components retained: {pca.n_components_}")
    
    return df_pca, pca

# Apply PCA to Cylinder Bottom Saw data
saw_df_pca, pca_saw = apply_pca(saw_df_filtered, n_components=0.95)

# Apply PCA to CNC Lathe data
lathe_df_pca, pca_lathe = apply_pca(lathe_df_filtered, n_components=0.95)

# Add the 'anomaly' column back to the PCA-transformed DataFrames
saw_df_pca["anomaly"] = saw_df["anomaly"].values
lathe_df_pca["anomaly"] = lathe_df["anomaly"].values

# ==============================================================================
# 🔍 Visualize PCA Components
# ==============================================================================
# Scatter plot of the first two principal components for Cylinder Bottom Saw
plt.figure(figsize=(10, 6))
plt.scatter(saw_df_pca.iloc[:, 0], saw_df_pca.iloc[:, 1], c=saw_df_pca["anomaly"], cmap="viridis", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cylinder Bottom Saw - First Two Principal Components")
plt.colorbar(label="Anomaly")
plt.show()

# Scatter plot of the first two principal components for CNC Lathe
plt.figure(figsize=(10, 6))
plt.scatter(lathe_df_pca.iloc[:, 0], lathe_df_pca.iloc[:, 1], c=lathe_df_pca["anomaly"], cmap="viridis", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("CNC Lathe - First Two Principal Components")
plt.colorbar(label="Anomaly")
plt.show()

# ==============================================================================
# 🔍 Autoencoder for Unsupervised Anomaly Detection
# ==============================================================================

# Check the 'anomaly' column in the original dataset
print("Cylinder Bottom Saw - Anomaly Distribution:")
print(saw_df["anomaly"].value_counts())

print("\nCNC Lathe - Anomaly Distribution:")
print(lathe_df["anomaly"].value_counts())

#==============================================================================
# ✅ Convert Non-Binary 'anomaly' Column to Binary (0 and 1)

# Convert non-binary 'anomaly' column to binary (0 and 1)
threshold = 0.5  # Adjust this threshold based on your data
saw_df["anomaly"] = (saw_df["anomaly"] > threshold).astype(int)
lathe_df["anomaly"] = (lathe_df["anomaly"] > threshold).astype(int)

# Verify the distribution after relabeling
print("Cylinder Bottom Saw - Anomaly Distribution (After Relabeling):")
print(saw_df["anomaly"].value_counts())

print("\nCNC Lathe - Anomaly Distribution (After Relabeling):")
print(lathe_df["anomaly"].value_counts())

#==============================================================================
# ✅ Split Data into Training and Testing Sets

# Split data into training (normal) and testing (normal + anomalies)
normal_data = saw_df_pca[saw_df_pca["anomaly"] == 0].drop(columns=["anomaly"])
anomalous_data = saw_df_pca[saw_df_pca["anomaly"] == 1].drop(columns=["anomaly"])

# Check if normal_data is empty
if len(normal_data) == 0:
    raise ValueError("No normal data found. Ensure the 'anomaly' column contains 0s (normal data).")

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(normal_data, test_size=0.2, random_state=42)
X_test = np.concatenate([X_test, anomalous_data])  # Add anomalies to the test set

# Build the autoencoder
input_dim = X_train.shape[1]
encoding_dim = 10  # Size of the bottleneck layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer="adam", loss="mse")

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Calculate reconstruction errors
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # 95th percentile of reconstruction errors

# Classify anomalies
y_pred = (mse > threshold).astype(int)

# Evaluate the model
y_true = np.concatenate([np.zeros(len(X_test) - len(anomalous_data)), np.ones(len(anomalous_data))])
print("Autoencoder Results:")
print(f"Precision: {precision_score(y_true, y_pred)}")
print(f"Recall: {recall_score(y_true, y_pred)}")
print(f"F1-Score: {f1_score(y_true, y_pred)}")
























# ==============================================================================
# ✅ Apply PCA for Dimensionality Reduction Only to the Feature Data
#  PCA is applied only to the scaled feature data 
# (saw_df_scaled and lathe_df_scaled), which does not include the anomaly column.
# ==============================================================================

#  Separate Features and Labels Before PCA
"""Before applying PCA, explicitly separate the features (independent variables) 
from the labels (anomaly  , time column)."""


# Drop the 'timestamp' and anomaly column from the feature data
X_saw = saw_df.drop(columns=["timestamp", "anomaly"])  # Features for Cylinder Bottom Saw
y_saw = saw_df["anomaly"]                             # Labels for Cylinder Bottom Saw

X_lathe = lathe_df.drop(columns=["timestamp", "anomaly"])  # Features for CNC Lathe
y_lathe = lathe_df["anomaly"]                              # Labels for CNC Lathe

#---------------------------------------------------------

"""Apply PCA Only to the Features
Apply PCA to the feature data (X_saw and X_lathe), not the labels (y_saw and y_lathe)."""

from sklearn.decomposition import PCA

# Apply PCA to Cylinder Bottom Saw features
pca_saw = PCA(n_components=0.95)  # Retain 95% of the variance
X_saw_pca = pca_saw.fit_transform(X_saw)

# Apply PCA to CNC Lathe features
pca_lathe = PCA(n_components=0.95)  # Retain 95% of the variance
X_lathe_pca = pca_lathe.fit_transform(X_lathe)

# Convert PCA-transformed data back to DataFrames
saw_df_pca = pd.DataFrame(X_saw_pca, columns=[f"PC{i+1}" for i in range(X_saw_pca.shape[1])])
lathe_df_pca = pd.DataFrame(X_lathe_pca, columns=[f"PC{i+1}" for i in range(X_lathe_pca.shape[1])])

# Add the 'anomaly' column back to the PCA-transformed DataFrames
saw_df_pca["anomaly"] = y_saw.values
lathe_df_pca["anomaly"] = y_lathe.values

#---------------------------------------------------------
"""Check the distribution of the anomaly column in the PCA-transformed DataFrames
to ensure it contains binary values (0 and 1)"""

# Check the 'anomaly' column in the PCA-transformed DataFrames
print("Cylinder Bottom Saw - Anomaly Distribution:")
print(saw_df_pca["anomaly"].value_counts())

print("\nCNC Lathe - Anomaly Distribution:")
print(lathe_df_pca["anomaly"].value_counts())

# Inspect the columns of the PCA-transformed DataFrames
print("Cylinder Bottom Saw PCA Columns:")
print(saw_df_pca.columns)

print("\nCNC Lathe PCA Columns:")
print(lathe_df_pca.columns)

#---------------------------------------------------------
# ✅ After PCA, the anomaly column is added back to the PCA-transformed DataFrames 
# (saw_df_pca and lathe_df_pca). This ensures that the labels are preserved for downstream tasks like visualization and model evaluation.

# Add the 'anomaly' column back to the PCA-transformed DataFrames
saw_df_pca["anomaly"] = saw_df["anomaly"].values
lathe_df_pca["anomaly"] = lathe_df["anomaly"].values


# ✅ Verification of Exclusion
"""To confirm that the anomaly column is excluded from PCA, 
you can inspect the columns of the PCA-transformed DataFrames before 
and after adding the anomaly column."""

# Inspect the columns of the PCA-transformed DataFrames before adding the 'anomaly' column
print("Cylinder Bottom Saw PCA Columns (Before Adding 'anomaly'):")
print(saw_df_pca.columns)

print("\nCNC Lathe PCA Columns (Before Adding 'anomaly'):")
print(lathe_df_pca.columns)


# Inspect the columns of the PCA-transformed DataFrames after adding the 'anomaly' column
print("Cylinder Bottom Saw PCA Columns (After Adding 'anomaly'):")
print(saw_df_pca.columns)

print("\nCNC Lathe PCA Columns (After Adding 'anomaly'):")
print(lathe_df_pca.columns)
#---------------------------------------------------------

print("Cylinder Bottom Saw (PCA-transformed):", saw_df_pca.shape)
print("CNC Lathe (PCA-transformed):", lathe_df_pca.shape)



#---------------------------------------------------------
# ✅ 2. Distribution of Labels Before and After PCA

"""Visualizing the feature space before and after PCA helps you 
to understand how the data has been transformed and whether the separation
between normal and anomalous data points has improved."""

#---------------------------------------------------------
# ==============================================================================
# 🔍 Visualize Label Distribution and Feature Space Before and After PCA
# ==============================================================================


# 1. Label Distribution Before PCA
plt.figure(figsize=(8, 5))
sns.countplot(x="anomaly", data=saw_df)
plt.title("Cylinder Bottom Saw - Label Distribution (Before PCA)")
plt.xlabel("Anomaly")
plt.ylabel("Count")
plt.show()

# 2. Feature Space Before PCA (First Two Features)
plt.figure(figsize=(10, 6))
plt.scatter(saw_df.iloc[:, 0], saw_df.iloc[:, 1], c=saw_df["anomaly"], cmap="viridis", alpha=0.5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Cylinder Bottom Saw - Feature Space (Before PCA)")
plt.colorbar(label="Anomaly")
plt.show()

#==============================================================================

# 3. Label Distribution After PCA
saw_df_pca["anomaly"] = saw_df["anomaly"]  # Add labels to PCA-transformed DataFrame
plt.figure(figsize=(8, 5))
sns.countplot(x="anomaly", data=saw_df_pca)
plt.title("Cylinder Bottom Saw - Label Distribution (After PCA)")
plt.xlabel("Anomaly")
plt.ylabel("Count")
plt.show()

# 4. Feature Space After PCA (First Two Principal Components)
plt.figure(figsize=(10, 6))
plt.scatter(saw_df_pca.iloc[:, 0], saw_df_pca.iloc[:, 1], c=saw_df_pca["anomaly"], cmap="viridis", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cylinder Bottom Saw - Feature Space (After PCA)")
plt.colorbar(label="Anomaly")
plt.show()


""" Cylinder Bottom Saw:

Retained 20 principal components to explain 95% of the variance.

CNC Lathe:

Retained 43 principal components to explain 95% of the variance.

These principal components are the transformed features after applying PCA. 
"""
# ==============================================================================
# 🔍 Visualize PCA Components
# ==============================================================================
# Scatter plot of the first two principal components for Cylinder Bottom Saw
plt.figure(figsize=(10, 6))
plt.scatter(saw_df_pca.iloc[:, 0], saw_df_pca.iloc[:, 1], c=saw_df_pca["anomaly"], cmap="viridis", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cylinder Bottom Saw - First Two Principal Components")
plt.colorbar(label="Anomaly")
plt.show()

# Scatter plot of the first two principal components for CNC Lathe
plt.figure(figsize=(10, 6))
plt.scatter(lathe_df_pca.iloc[:, 0], lathe_df_pca.iloc[:, 1], c=lathe_df_pca["anomaly"], cmap="viridis", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("CNC Lathe - First Two Principal Components")
plt.colorbar(label="Anomaly")
plt.show()


# ==============================================================================
# 🔍 Add 'anomaly' Column to PCA-Transformed DataFrames
# ==============================================================================

# Add the 'anomaly' column to the PCA-transformed DataFrames
lathe_df_pca["anomaly"] = lathe_df["anomaly"]
saw_df_pca["anomaly"] = saw_df["anomaly"]





import plotly.express as px

# Interactive 3D Scatter Plot for Cylinder Bottom Saw
fig = px.scatter_3d(
    saw_df_pca,
    x="PC1",
    y="PC2",
    z="PC3",
    color="anomaly",
    title="Cylinder Bottom Saw - First Three Principal Components",
    labels={"anomaly": "Anomaly"},
    opacity=0.7,
)
fig.show()

# Interactive 3D Scatter Plot for CNC Lathe
fig = px.scatter_3d(
    lathe_df_pca,
    x="PC1",
    y="PC2",
    z="PC3",
    color="anomaly",
    title="CNC Lathe - First Three Principal Components",
    labels={"anomaly": "Anomaly"},
    opacity=0.7,
)
fig.show()

# ==============================================================================
# 🔍 Save PCA Models in Multiple Formats
# ==============================================================================
# This script demonstrates how to save PCA models in various formats:
# 1. Pickle (.pkl) - Python's native serialization format.
# 2. joblib (.joblib) - Optimized for large NumPy arrays.
# 3. HDF5 (.h5) - Hierarchical Data Format, commonly used for deep learning models.
# 4. ONNX (.onnx) - Open Neural Network Exchange, for interoperability.
# 5. PMML (.pmml) - Predictive Model Markup Language, for cross-platform sharing.
# ==============================================================================

import joblib
import h5py
import onnx
import onnxmltools
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import pickle

# ==============================================================================
# ✅ 1. Save PCA Models Using Pickle
# ==============================================================================
# Pickle is Python's native serialization format. It's simple but has security risks.
# Use it for quick saving and loading within trusted environments.

# Save PCA models
with open("pca_saw_model.pkl", "wb") as f:
    pickle.dump(pca_saw, f)

with open("pca_lathe_model.pkl", "wb") as f:
    pickle.dump(pca_lathe, f)

# Load PCA models
with open("pca_saw_model.pkl", "rb") as f:
    pca_saw_loaded = pickle.load(f)

with open("pca_lathe_model.pkl", "rb") as f:
    pca_lathe_loaded = pickle.load(f)

print("✅ PCA models saved and loaded using Pickle.")

# ==============================================================================
# ✅ 2. Save PCA Models Using joblib
# ==============================================================================
# joblib is optimized for large NumPy arrays and is faster than Pickle for scikit-learn models.

# Save PCA models
joblib.dump(pca_saw, "pca_saw_model.joblib")
joblib.dump(pca_lathe, "pca_lathe_model.joblib")

# Load PCA models
pca_saw_loaded = joblib.load("pca_saw_model.joblib")
pca_lathe_loaded = joblib.load("pca_lathe_model.joblib")

print("✅ PCA models saved and loaded using joblib.")

# ==============================================================================
# ✅ 3. Save PCA Models Using HDF5
# ==============================================================================
# HDF5 is a hierarchical data format commonly used for deep learning models.
# It supports large datasets and is efficient for saving model weights.

# Save PCA models
with h5py.File("pca_saw_model.h5", "w") as f:
    f.create_dataset("components_", data=pca_saw.components_)
    f.create_dataset("explained_variance_", data=pca_saw.explained_variance_)
    f.create_dataset("mean_", data=pca_saw.mean_)

with h5py.File("pca_lathe_model.h5", "w") as f:
    f.create_dataset("components_", data=pca_lathe.components_)
    f.create_dataset("explained_variance_", data=pca_lathe.explained_variance_)
    f.create_dataset("mean_", data=pca_lathe.mean_)

# Load PCA models
with h5py.File("pca_saw_model.h5", "r") as f:
    components_ = f["components_"][:]
    explained_variance_ = f["explained_variance_"][:]
    mean_ = f["mean_"][:]

pca_saw_loaded = PCA()
pca_saw_loaded.components_ = components_
pca_saw_loaded.explained_variance_ = explained_variance_
pca_saw_loaded.mean_ = mean_

with h5py.File("pca_lathe_model.h5", "r") as f:
    components_ = f["components_"][:]
    explained_variance_ = f["explained_variance_"][:]
    mean_ = f["mean_"][:]

pca_lathe_loaded = PCA()
pca_lathe_loaded.components_ = components_
pca_lathe_loaded.explained_variance_ = explained_variance_
pca_lathe_loaded.mean_ = mean_

print("✅ PCA models saved and loaded using HDF5.")

# ==============================================================================
# ✅ 4. Save PCA Models Using ONNX
# ==============================================================================
# ONNX is a standardized format for interoperability across frameworks.
# It's ideal for deploying models in production environments.

# Convert PCA models to ONNX format
pca_saw_onnx = onnxmltools.convert_sklearn(pca_saw, "pca_saw_model.onnx")
pca_lathe_onnx = onnxmltools.convert_sklearn(pca_lathe, "pca_lathe_model.onnx")

# Save ONNX models
onnx.save_model(pca_saw_onnx, "pca_saw_model.onnx")
onnx.save_model(pca_lathe_onnx, "pca_lathe_model.onnx")

# Load ONNX models
import onnxruntime as ort

session_saw = ort.InferenceSession("pca_saw_model.onnx")
session_lathe = ort.InferenceSession("pca_lathe_model.onnx")

print("✅ PCA models saved and loaded using ONNX.")

# ==============================================================================
# ✅ 5. Save PCA Models Using PMML
# ==============================================================================
# PMML is an XML-based format for sharing models across platforms.
# It's supported by many machine learning libraries.

# Create PMML pipelines
pipeline_saw = PMMLPipeline([("pca", pca_saw)])
pipeline_lathe = PMMLPipeline([("pca", pca_lathe)])

# Save PCA models as PMML
sklearn2pmml(pipeline_saw, "pca_saw_model.pmml")
sklearn2pmml(pipeline_lathe, "pca_lathe_model.pmml")

print("✅ PCA models saved and loaded using PMML.")


# ==============================================================================

"""
let’s move on to the next phase of your Anomaly Detection Pipeline.
We’ll implement the following three approaches:

Autoencoder: Learns the normal data distribution and detects anomalies based on reconstruction errors.
Random Forest: Trained as a supervised classifier using labeled anomalies and normal data.
One-Class SVM: Models normal behavior in an unsupervised manner and detects anomalies as outliers.
"""

# ==============================================================================

#---------------------------------------------------------
# ✅ 1. Autoencoder for Unsupervised Anomaly Detection

"""
Objective: Train an autoencoder on normal data and use reconstruction error to detect anomalies.

Steps:
Split the data into training (normal data) and testing (normal + anomalies).
Build and train an autoencoder model.
Calculate reconstruction errors and set a threshold for anomaly detection.
Evaluate the model using metrics like precision, recall, and F1-score.

"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import precision_score, recall_score, f1_score



# Check the 'anomaly' column in the PCA-transformed DataFrame
"""First, check if the anomaly column exists in the PCA-transformed DataFrame 
and contains both 0 (normal) and 1 (anomaly) values."""

print("Cylinder Bottom Saw - Anomaly Distribution:")
print(saw_df_pca["anomaly"].value_counts())

print("\nCNC Lathe - Anomaly Distribution:")
print(lathe_df_pca["anomaly"].value_counts())

#---------------------------------------------------------
# Split data into training (normal) and testing (normal + anomalies)
normal_data = saw_df_pca[saw_df_pca["anomaly"] == 0].drop(columns=["anomaly"])
anomalous_data = saw_df_pca[saw_df_pca["anomaly"] == 1].drop(columns=["anomaly"])

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(normal_data, test_size=0.2, random_state=42)
X_test = np.concatenate([X_test, anomalous_data])  # Add anomalies to the test set

# Build the autoencoder
input_dim = X_train.shape[1]
encoding_dim = 10  # Size of the bottleneck layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer="adam", loss="mse")

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Calculate reconstruction errors
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # 95th percentile of reconstruction errors

# Classify anomalies
y_pred = (mse > threshold).astype(int)

# Evaluate the model
y_true = np.concatenate([np.zeros(len(X_test) - len(anomalous_data)), np.ones(len(anomalous_data))])
print("Autoencoder Results:")
print(f"Precision: {precision_score(y_true, y_pred)}")
print(f"Recall: {recall_score(y_true, y_pred)}")
print(f"F1-Score: {f1_score(y_true, y_pred)}")

#---------------------------------------------------------











































































# =====================================================================
# ✅ Apply PCA for Dimensionality Reduction
# =====================================================================
def apply_pca(df, n_components=3):
    """Applies PCA for dimensionality reduction."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

    return pca, reduced_data

# Apply PCA to CNC Lathe Data
pca, reduced_data = apply_pca(lathe_df)

# =====================================================================
# 📊 3D Scatter Plot of PCA Components
# =====================================================================
def plot_pca_3d(data, title):
    """Creates a 3D scatter plot of the first 3 PCA components."""
    fig = px.scatter_3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2], 
        title=f"{title} - 3D PCA Scatter Plot",
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
        opacity=0.8
    )
    fig.show()

plot_pca_3d(reduced_data, "CNC Lathe")

# =====================================================================
# 🔍 Autoencoder Reconstruction Error
# =====================================================================
# Load Pretrained Autoencoder
autoencoder_path = "data/models/cnc_lathe_autoencoder.h5"
autoencoder = load_model(autoencoder_path)

# Compute Reconstruction Error
def compute_reconstruction_error(autoencoder, data):
    """Computes reconstruction error for given data using autoencoder."""
    reconstructed_data = autoencoder.predict(data)
    errors = np.mean(np.square(data - reconstructed_data), axis=1)
    return errors

# Compute errors
reconstruction_errors = compute_reconstruction_error(autoencoder, reduced_data)

# =====================================================================
# 📉 Distribution of Reconstruction Errors
# =====================================================================
def plot_reconstruction_error(errors, title):
    """Plots a KDE distribution of reconstruction errors."""
    plt.figure(figsize=(10, 5))
    sns.kdeplot(errors, fill=True, color="red", alpha=0.6)
    plt.title(f"{title} - Distribution of Reconstruction Errors")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.show()

plot_reconstruction_error(reconstruction_errors, "CNC Lathe")

# =====================================================================
# ✅ Random Forest for Classification
# =====================================================================
# Prepare Labels
labels = (reconstruction_errors > np.percentile(reconstruction_errors, 95)).astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    reconstruction_errors.reshape(-1, 1), labels, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# =====================================================================
# 📊 Confusion Matrix Visualization
# =====================================================================
def plot_confusion_matrix(y_test, y_pred, title):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{title} - Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_test, y_pred, "Random Forest")

# =====================================================================
# 🏆 Feature Importance Plot
# =====================================================================
def plot_feature_importance(model, title):
    """Plots feature importance from a trained model."""
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    sns.barplot(x=importance, y=["Reconstruction Error"], palette="viridis")
    plt.title(f"{title} - Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

plot_feature_importance(rf_classifier, "Random Forest")

# =====================================================================
# 📦 Box Plot for Normal vs. Anomalous Data
# =====================================================================
def plot_boxplot(errors, labels, title):
    """Creates a box plot for reconstruction errors of normal and anomalous data."""
    df = pd.DataFrame({"Reconstruction Error": errors, "Label": labels})
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Label", y="Reconstruction Error", data=df, palette="coolwarm")
    plt.title(f"{title} - Box Plot of Reconstruction Errors")
    plt.xticks(ticks=[0, 1], labels=["Normal", "Anomalous"])
    plt.show()

plot_boxplot(reconstruction_errors, labels, "CNC Lathe")

# =====================================================================
# 🎯 Interactive ROC Curve
# =====================================================================
def plot_roc_curve(y_test, y_pred, title):
    """Creates an interactive ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f"{title} - ROC Curve (AUC = {auc_score:.3f})",
                  labels=dict(x="False Positive Rate", y="True Positive Rate"))
    fig.show()

plot_roc_curve(y_test, y_pred, "Random Forest")

# =====================================================================
# ✅ Anomaly Detection Pipeline Completed
# =====================================================================
print("🚀 Anomaly Detection Pipeline Completed with Enhanced Visualizations!")
