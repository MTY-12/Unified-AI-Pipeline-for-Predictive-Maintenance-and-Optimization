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
cylinder_bottom_saw_df.head()
cylinder_bottom_saw_df.tail()
piston_rod_cnc_lathe_df.head()
piston_rod_cnc_lathe_df.tail()

#===============================================================================

# ==============================================================================
# ✅ Feature Scaling - Standardization

"""Step 1: It selects only numerical columns using df.select_dtypes(include=[np.number]).columns
Step 2: It applies StandardScaler only to those numeric columns.
Step 3: timestamp and anomaly are ignored because they are not numerical features."""
# ==============================================================================

# ==============================================================================
# ✅ Feature Scaling - Standardization
# ==============================================================================
from sklearn.preprocessing import StandardScaler

# ============================================================================== 
# ✅ Feature Scaling - Standardization (Fixed)
# ==============================================================================

# ==============================================================================
# ✅ Feature Scaling - Standardization (Final Fix)
# ==============================================================================

# ==============================================================================
# ✅ Feature Scaling - Standardization (Final Fix)
# ==============================================================================

def scale_features(df):
    """Applies standard scaling to numerical features, explicitly preserving 'timestamp' and binarizing 'anomaly'."""
    
    scaler = StandardScaler()

    # Make a copy to avoid modifying the original dataframe
    df_scaled = df.copy()

    # Exclude 'timestamp' and 'anomaly' from scaling
    exclude_cols = ['timestamp', 'anomaly']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    # Apply StandardScaler only to the selected numeric columns
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # Ensure 'anomaly' column is strictly binary (0 or 1)
    df_scaled['anomaly'] = df['anomaly'].apply(lambda x: 1 if x > 0 else 0)

    logging.info(f"✅ Data scaled successfully. {len(numeric_cols)} features standardized.")
    return df_scaled, scaler

# Apply corrected scaling function
cylinder_bottom_saw_df, saw_scaler = scale_features(cylinder_bottom_saw_df)
piston_rod_cnc_lathe_df, lathe_scaler = scale_features(piston_rod_cnc_lathe_df)

# # Save the scalers for future use
# joblib.dump(saw_scaler, "models/saw_scaler.pkl")
# joblib.dump(lathe_scaler, "models/lathe_scaler.pkl")

# Verify the fix
print(cylinder_bottom_saw_df[['timestamp', 'anomaly']].tail())
print(piston_rod_cnc_lathe_df[['timestamp', 'anomaly']].tail())


# Verify that 'timestamp' and 'anomaly' are unchanged
print(cylinder_bottom_saw_df[['timestamp', 'anomaly']].head())
print(piston_rod_cnc_lathe_df[['timestamp', 'anomaly']].head())
print(cylinder_bottom_saw_df[['timestamp', 'anomaly']].tail())
print(piston_rod_cnc_lathe_df[['timestamp', 'anomaly']].tail())

# Count the number of anomalies in each dataset

# Count anomalies for Cylinder Bottom Saw
saw_anomaly_count = cylinder_bottom_saw_df['anomaly'].value_counts()

# Count anomalies for CNC Lathe (Piston Rod)
lathe_anomaly_count = piston_rod_cnc_lathe_df['anomaly'].value_counts()

# Display results
anomaly_counts = pd.DataFrame({
    "Machine": ["Cylinder Bottom Saw", "CNC Lathe (Piston Rod)"],
    "Normal Count (0)": [saw_anomaly_count.get(0, 0), lathe_anomaly_count.get(0, 0)],
    "Anomaly Count (1)": [saw_anomaly_count.get(1, 0), lathe_anomaly_count.get(1, 0)]
})

print(anomaly_counts)




joblib.dump(saw_scaler, "saw_scaler.pkl.joblib")
joblib.dump(lathe_scaler, "lathe_scaler.pkl.joblib")
# ==============================================================================
# ✅ Apply PCA for Dimensionality Reduction

#  ✅ Goals of this step:
# 1️⃣ Train PCA only on normal data to retain essential patterns.
# 2️⃣ Apply trained PCA model to the entire dataset (normal + anomalies) for transformation.
# 3️⃣ Visualize PCA results to analyze normal vs. anomalous data distribution.
# ==============================================================================

# 🔹 Step 1: Train PCA on Normal Data Only
# We will train PCA using only normal instances (anomaly = 0) for each machine.
# ==============================================================================    
# ✅ Apply PCA for Dimensionality Reduction (Trained on Normal Data Only)
# ==============================================================================  

def apply_pca(df, variance_threshold=0.95):
    """
    Applies PCA on the normal data and transforms the entire dataset.
    
    Args:
        df (pd.DataFrame): The dataset containing all instances (normal + anomalies).
        variance_threshold (float): The amount of variance to retain in PCA.
    
    Returns:
        tuple: (PCA object, PCA-transformed DataFrame)
    """
    # Separate normal data (anomaly = 0)
    normal_data = df[df["anomaly"] == 0].drop(columns=["anomaly", "timestamp"], errors="ignore")
    
    # Standardize the features before applying PCA
    scaler = StandardScaler()
    normal_data_scaled = scaler.fit_transform(normal_data)
    
    # Apply PCA
    pca = PCA(n_components=variance_threshold)
    normal_pca_transformed = pca.fit_transform(normal_data_scaled)
    
    # Apply trained PCA transformation to the full dataset (including anomalies)
    full_data_scaled = scaler.transform(df.drop(columns=["anomaly", "timestamp"], errors="ignore"))
    full_pca_transformed = pca.transform(full_data_scaled)
    
    # Convert transformed data into a DataFrame
    pca_columns = [f"PC{i+1}" for i in range(full_pca_transformed.shape[1])]
    pca_df = pd.DataFrame(full_pca_transformed, columns=pca_columns)
    
    # Add back anomaly labels and timestamps
    pca_df["timestamp"] = df["timestamp"].values
    pca_df["anomaly"] = df["anomaly"].values
    
    # Print explained variance
    explained_variance = sum(pca.explained_variance_ratio_)
    logging.info(f"✅ PCA applied. Retained Variance: {explained_variance:.4f}")
    
    return pca, pca_df, scaler

# ==============================================================================  
# ✅ Train PCA for Cylinder Bottom Saw
pca_saw, saw_pca_df, saw_scaler = apply_pca(cylinder_bottom_saw_df)
# ✅ Train PCA for CNC Lathe
pca_lathe, lathe_pca_df, lathe_scaler = apply_pca(piston_rod_cnc_lathe_df)

# ==============================================================================  

# ✅ Save PCA Models and Scalers for Future Use
joblib.dump(pca_saw, "saw_pca_model.pkl")
joblib.dump(saw_scaler, "saw_scaler.pkl")
joblib.dump(pca_lathe, "lathe_pca_model.pkl")
joblib.dump(lathe_scaler, "lathe_scaler.pkl")


# ==============================================================================  
# ✅ 2D Scatter Plot for PCA-Transformed Data

# 🔹 Step 2: Visualize PCA-Transformed Data
# ==============================================================================  
def plot_pca_2d(pca_df, title):
    """
    Plots a 2D scatter plot of the first two principal components.
    
    Args:
        pca_df (pd.DataFrame): PCA-transformed DataFrame containing "PC1", "PC2", and "anomaly" columns.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], hue=pca_df["anomaly"], palette={0: "blue", 1: "red"}, alpha=0.5)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Anomaly", labels=["Normal", "Anomaly"])
    plt.show()

# Plot for Cylinder Bottom Saw
plot_pca_2d(saw_pca_df, "Cylinder Bottom Saw - PCA (2D Visualization)")

# Plot for CNC Lathe
plot_pca_2d(lathe_pca_df, "CNC Lathe (Piston Rod) - PCA (2D Visualization)")


#-------------------------------------------

# ==============================================================================  
# ✅ 3D Scatter Plot for PCA-Transformed Data
# ==============================================================================  
def plot_pca_3d(pca_df, title):
    """
    Plots a 3D scatter plot of the first three principal components.
    
    Args:
        pca_df (pd.DataFrame): PCA-transformed DataFrame containing "PC1", "PC2", "PC3", and "anomaly" columns.
        title (str): Title of the plot.
    """
    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=pca_df["anomaly"].astype(str),
                        color_discrete_map={"0": "blue", "1": "red"},
                        title=title, opacity=0.7)
    fig.show()

# Plot for Cylinder Bottom Saw
plot_pca_3d(saw_pca_df, "Cylinder Bottom Saw - PCA (3D Visualization)")

# Plot for CNC Lathe
plot_pca_3d(lathe_pca_df, "CNC Lathe (Piston Rod) - PCA (3D Visualization)")

#-------------------------------------------

# ==============================================================================

# 🚀  Train an Autoencoder for Anomaly Detection
# Now that we have PCA-transformed data, the next step is to train Autoencoders for detecting anomalies.
"""
✅ Goals of this Step:
1️⃣ Build an Autoencoder neural network for each machine (Saw & CNC Lathe).
2️⃣ Train the Autoencoder only on normal data (anomaly = 0).
3️⃣ Evaluate reconstruction errors on both normal and anomalous data.
4️⃣ Set an anomaly detection threshold based on reconstruction error."""
# ==============================================================================
# 🔹 Step 1: Define the Autoencoder Model

# ==============================================================================  
# ✅ Define Autoencoder Model
# ==============================================================================  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    """
    Constructs an Autoencoder Model with an Encoder-Decoder architecture.

    Args:
        input_dim (int): Number of input features (PCA components).

    Returns:
        Model: Compiled Autoencoder model.
    """

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(input_layer)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(8, activation="relu")(encoded)

    # Decoder
    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(decoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)

    # Autoencoder Model
    autoencoder = Model(input_layer, output_layer)

    # Compile Model
    autoencoder.compile(optimizer="adam", loss="mse")
    
    return autoencoder

#-------------------------------------------
# 🔹 Step 2: Train Autoencoder on Normal Data

"""train the Autoencoder only on normal data (anomaly = 0).
After training, we evaluate the reconstruction error to detect anomalies."""

# ==============================================================================  
# ✅ Train Autoencoder on Normal Data
# ==============================================================================  

# Extract only normal data for training
saw_train = saw_pca_df[saw_pca_df["anomaly"] == 0].drop(columns=["anomaly", "timestamp"], errors="ignore")
lathe_train = lathe_pca_df[lathe_pca_df["anomaly"] == 0].drop(columns=["anomaly", "timestamp"], errors="ignore")

# Get input dimensions
saw_input_dim = saw_train.shape[1]
lathe_input_dim = lathe_train.shape[1]

# Build autoencoder models
saw_autoencoder = build_autoencoder(saw_input_dim)
lathe_autoencoder = build_autoencoder(lathe_input_dim)

# Train the autoencoders
EPOCHS = 50
BATCH_SIZE = 32

print("🚀 Training Autoencoder for Cylinder Bottom Saw...")
saw_autoencoder.fit(saw_train, saw_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1, verbose=1)

print("🚀 Training Autoencoder for CNC Lathe...")
lathe_autoencoder.fit(lathe_train, lathe_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1, verbose=1)

 
 # ============================================================================== 
 # 🔹 Step 3: Compute Reconstruction Error
""" Reconstruction error measures how well the Autoencoder reconstructs normal vs. anomalous data.
 Higher reconstruction error = Likely anomaly.""" 
# ✅ Compute Reconstruction Error
# ==============================================================================  
def compute_reconstruction_error(autoencoder, df):
    """
    Computes the reconstruction error for given data using the trained Autoencoder.

    Args:
        autoencoder (Model): Trained Autoencoder model.
        df (DataFrame): PCA-transformed data.

    Returns:
        np.array: Reconstruction errors.
    """
    reconstructed_data = autoencoder.predict(df)
    errors = np.mean(np.square(df - reconstructed_data), axis=1)
    return errors

# Compute reconstruction errors for full datasets
saw_reconstruction_errors = compute_reconstruction_error(saw_autoencoder, saw_pca_df.drop(columns=["anomaly", "timestamp"], errors="ignore"))
lathe_reconstruction_errors = compute_reconstruction_error(lathe_autoencoder, lathe_pca_df.drop(columns=["anomaly", "timestamp"], errors="ignore"))

# Add reconstruction errors to DataFrames
saw_pca_df["reconstruction_error"] = saw_reconstruction_errors
lathe_pca_df["reconstruction_error"] = lathe_reconstruction_errors

#===============================================================================

# 🔹 Step 4: Define Anomaly Detection Threshold
"""To detect anomalies, we need to set a threshold based 
on the 95th percentile of normal reconstruction errors."""

# ✅ Define Anomaly Detection Threshold
# ==============================================================================  
# Define a static anomaly threshold (95th percentile)
saw_threshold = np.percentile(saw_reconstruction_errors[saw_pca_df["anomaly"] == 0], 95)
lathe_threshold = np.percentile(lathe_reconstruction_errors[lathe_pca_df["anomaly"] == 0], 95)

# Classify anomalies based on threshold
saw_pca_df["predicted_anomaly"] = (saw_pca_df["reconstruction_error"] > saw_threshold).astype(int)
lathe_pca_df["predicted_anomaly"] = (lathe_pca_df["reconstruction_error"] > lathe_threshold).astype(int)

#===============================================================================
# 🔹 Step 5: Evaluate Anomaly Detection Performance
"""We compare actual vs. predicted anomalies using Precision, Recall, and F1-score.""" 
# ✅ Evaluate Anomaly Detection Performance
# ==============================================================================  
from sklearn.metrics import classification_report, confusion_matrix

print("🔹 Cylinder Bottom Saw - Autoencoder Performance")
print(classification_report(saw_pca_df["anomaly"], saw_pca_df["predicted_anomaly"]))

print("\n🔹 CNC Lathe - Autoencoder Performance")
print(classification_report(lathe_pca_df["anomaly"], lathe_pca_df["predicted_anomaly"]))

#===============================================================================
# 🔹 Step 6: Visualize Reconstruction Errors
# A Histogram helps visualize how well normal vs. anomalous samples are separated.
# ==============================================================================  
# ✅ Visualizing Reconstruction Errors
# ==============================================================================  
def plot_reconstruction_error(df, threshold, title):
    """
    Plots a histogram of reconstruction errors.
    
    Args:
        df (DataFrame): The dataset containing reconstruction errors and anomaly labels.
        threshold (float): The anomaly detection threshold.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(df[df["anomaly"] == 0]["reconstruction_error"], bins=50, alpha=0.7, color="blue", label="Normal Data")
    plt.hist(df[df["anomaly"] == 1]["reconstruction_error"], bins=50, alpha=0.7, color="red", label="Anomalous Data")
    plt.axvline(threshold, color="black", linestyle="dashed", linewidth=2, label="Anomaly Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.show()

# Plot for Cylinder Bottom Saw
plot_reconstruction_error(saw_pca_df, saw_threshold, "Cylinder Bottom Saw - Reconstruction Error Distribution")

# Plot for CNC Lathe
plot_reconstruction_error(lathe_pca_df, lathe_threshold, "CNC Lathe (Piston Rod) - Reconstruction Error Distribution")




#===============================================================================
#===============================================================================
"""The current issue is that the Autoencoder fails to detect many anomalies (low recall for class 1). This happens because the chosen threshold for reconstruction error is too high, allowing some anomalies to be misclassified as normal.

✅ Solution: Adjust the Threshold Dynamically
Instead of using a fixed 95th percentile threshold, we will:

Test different percentile thresholds (e.g., 90%, 92.5%, 95%, 97.5%, 99%) to find the best trade-off between precision and recall.
Analyze the Precision-Recall trade-off using a Precision-Recall Curve.
Select an optimized threshold that improves recall without excessive false positives. """

#------------------------------------------------------------------------------------------------
#  Step 1: Try Different Percentile Thresholds
# Let's evaluate multiple thresholds and compare their performance.
from sklearn.metrics import classification_report

def evaluate_thresholds(df, errors, actual_labels, percentiles=[90, 92.5, 95, 97.5, 99]):
    """
    Tests different reconstruction error thresholds and prints classification reports.

    Args:
        df (DataFrame): The dataset containing reconstruction errors.
        errors (np.array): Computed reconstruction errors.
        actual_labels (pd.Series): Ground truth anomaly labels.
        percentiles (list): List of percentile thresholds to test.

    Returns:
        dict: Best threshold and classification report.
    """
    best_f1 = 0
    best_threshold = None
    best_report = None

    for p in percentiles:
        threshold = np.percentile(errors, p)
        df["adjusted_predicted_anomaly"] = (errors > threshold).astype(int)

        report = classification_report(actual_labels, df["adjusted_predicted_anomaly"], output_dict=True)
        f1_score = report["1"]["f1-score"]  # Get F1-score for anomalies (Class 1)

        print(f"🔹 Threshold at {p}th percentile (Error > {threshold:.4f}):\n")
        print(classification_report(actual_labels, df["adjusted_predicted_anomaly"]))

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
            best_report = report

    return {"best_threshold": best_threshold, "best_report": best_report}

#-------------------------------------------
# 🔥 Step 2: Apply the Function to Your Data
# 2️⃣ Evaluate the new thresholds for both machines

# Cylinder Bottom Saw
print("\n🔎 Evaluating Thresholds for Cylinder Bottom Saw:")
saw_threshold_results = evaluate_thresholds(saw_pca_df, saw_reconstruction_errors, saw_pca_df["anomaly"])

# CNC Lathe (Piston Rod)
print("\n🔎 Evaluating Thresholds for CNC Lathe:")
lathe_threshold_results = evaluate_thresholds(lathe_pca_df, lathe_reconstruction_errors, lathe_pca_df["anomaly"])

#-------------------------------------------

🔥 Step 3: Select the Best Threshold & Apply It
3️⃣ Set the new optimized threshold

# Get best thresholds from evaluation results
saw_best_threshold = saw_threshold_results["best_threshold"]
lathe_best_threshold = lathe_threshold_results["best_threshold"]

# Apply best threshold to classify anomalies
saw_pca_df["optimized_predicted_anomaly"] = (saw_reconstruction_errors > saw_best_threshold).astype(int)
lathe_pca_df["optimized_predicted_anomaly"] = (lathe_reconstruction_errors > lathe_best_threshold).astype(int)

# Final Evaluation
print("\n🔎 Final Performance for Cylinder Bottom Saw (Optimized Threshold):")
print(classification_report(saw_pca_df["anomaly"], saw_pca_df["optimized_predicted_anomaly"]))

print("\n🔎 Final Performance for CNC Lathe (Optimized Threshold):")
print(classification_report(lathe_pca_df["anomaly"], lathe_pca_df["optimized_predicted_anomaly"]))
#------------------------------------------------------------------------------------------------

# 🔥 Step 4: Precision-Recall Curve Visualization
# To ensure our threshold is well-chosen, let's plot the Precision-Recall Curve.


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_precision_recall(y_true, y_scores, title):
    """Plots the Precision-Recall Curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for Cylinder Bottom Saw
plot_precision_recall(saw_pca_df["anomaly"], saw_reconstruction_errors, "Cylinder Bottom Saw - Precision-Recall Curve")

# Plot for CNC Lathe
plot_precision_recall(lathe_pca_df["anomaly"], lathe_reconstruction_errors, "CNC Lathe - Precision-Recall Curve")



#----------------------------------------------------------------------------------------------


# Re-import necessary libraries after execution state reset

from sklearn.metrics import precision_recall_fscore_support

# Define different threshold percentiles to test
threshold_percentiles = [60, 70, 80, 90, 95]

# Simulated reconstruction errors and anomaly labels (replace with actual data)
np.random.seed(42)  # For reproducibility
saw_reconstruction_errors = np.random.gamma(shape=2, scale=1, size=1000)
saw_anomalies = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # Imbalanced data

lathe_reconstruction_errors = np.random.gamma(shape=2, scale=1, size=1000)
lathe_anomalies = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

# Initialize lists to store results
precision_scores_saw, recall_scores_saw, f1_scores_saw = [], [], []
precision_scores_lathe, recall_scores_lathe, f1_scores_lathe = [], [], []

# Function to evaluate different thresholds
def evaluate_thresholds(reconstruction_errors, true_labels):
    precision_scores, recall_scores, f1_scores = [], [], []
    for percentile in threshold_percentiles:
        threshold = np.percentile(reconstruction_errors, percentile)
        predicted_labels = (reconstruction_errors > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary")

        # Store results
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return precision_scores, recall_scores, f1_scores

# Evaluate for Cylinder Bottom Saw
precision_scores_saw, recall_scores_saw, f1_scores_saw = evaluate_thresholds(saw_reconstruction_errors, saw_anomalies)

# Evaluate for CNC Lathe
precision_scores_lathe, recall_scores_lathe, f1_scores_lathe = evaluate_thresholds(lathe_reconstruction_errors, lathe_anomalies)

# Plot Precision, Recall, and F1-Score across different thresholds for Cylinder Bottom Saw
plt.figure(figsize=(10, 5))
plt.plot(threshold_percentiles, precision_scores_saw, marker="o", label="Precision", color="blue")
plt.plot(threshold_percentiles, recall_scores_saw, marker="o", label="Recall", color="red")
plt.plot(threshold_percentiles, f1_scores_saw, marker="o", label="F1-Score", color="green")

plt.xlabel("Threshold Percentile")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score vs. Threshold (Cylinder Bottom Saw)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Precision, Recall, and F1-Score across different thresholds for CNC Lathe
plt.figure(figsize=(10, 5))
plt.plot(threshold_percentiles, precision_scores_lathe, marker="o", label="Precision", color="blue")
plt.plot(threshold_percentiles, recall_scores_lathe, marker="o", label="Recall", color="red")
plt.plot(threshold_percentiles, f1_scores_lathe, marker="o", label="F1-Score", color="green")

plt.xlabel("Threshold Percentile")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score vs. Threshold (CNC Lathe)")
plt.legend()
plt.grid(True)
plt.show()












#===============================================================================
# 🔹 Step 3: Train Random Forest Classifier
# Since we have labeled anomalies, we can train a supervised Random Forest Classifier.

# ✅ Train Random Forest Classifier
# ==============================================================================  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
saw_X = saw_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error", "svm_predicted_anomaly"], errors="ignore")
saw_y = saw_pca_df["anomaly"]

lathe_X = lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error", "svm_predicted_anomaly"], errors="ignore")
lathe_y = lathe_pca_df["anomaly"]

# Train-Test Split
saw_X_train, saw_X_test, saw_y_train, saw_y_test = train_test_split(saw_X, saw_y, test_size=0.2, random_state=42)
lathe_X_train, lathe_X_test, lathe_y_train, lathe_y_test = train_test_split(lathe_X, lathe_y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
saw_rf = RandomForestClassifier(n_estimators=100, random_state=42)
lathe_rf = RandomForestClassifier(n_estimators=100, random_state=42)

saw_rf.fit(saw_X_train, saw_y_train)
lathe_rf.fit(lathe_X_train, lathe_y_train)

print("✅ Random Forest Classifier trained on labeled data.")
#-------------------------------------------

# 🔹 Step 4: Predict Anomalies using Random Forest


saw_pca_df["rf_predicted_anomaly"] = saw_rf.predict(saw_X)
lathe_pca_df["rf_predicted_anomaly"] = lathe_rf.predict(lathe_X)

# Evaluate Performance
print("🔹 Cylinder Bottom Saw - Random Forest Performance")
print(classification_report(saw_y, saw_pca_df["rf_predicted_anomaly"]))

print("\n🔹 CNC Lathe - Random Forest Performance")
print(classification_report(lathe_y, lathe_pca_df["rf_predicted_anomaly"]))



#-------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, machine_name):
    """
    Plots the confusion matrix for the given model predictions.

    Parameters:
    - y_true: Ground truth anomaly labels.
    - y_pred: Model predicted labels.
    - machine_name: Name of the machine (e.g., "Cylinder Bottom Saw", "CNC Lathe").
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix - Random Forest ({machine_name})")
    plt.show()

# Plot for Saw and Lathe
plot_confusion_matrix(saw_y, saw_pca_df["rf_predicted_anomaly"], "Cylinder Bottom Saw")
plot_confusion_matrix(lathe_y, lathe_pca_df["rf_predicted_anomaly"], "CNC Lathe")

#-------------------------------------------

import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names, machine_name):
    """
    Plots the feature importance derived from the Random Forest model.

    Parameters:
    - model: Trained Random Forest model.
    - feature_names: List of feature names.
    - machine_name: Name of the machine (e.g., "Cylinder Bottom Saw", "CNC Lathe").
    """
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="viridis")
    plt.title(f"Feature Importance - Random Forest ({machine_name})")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

# Plot for Saw and Lathe
plot_feature_importance(saw_rf, saw_X.columns, "Cylinder Bottom Saw")
plot_feature_importance(lathe_rf, lathe_X.columns, "CNC Lathe")

#-------------------------------------------
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names, machine_name):
    """
    Plots the feature importance derived from the Random Forest model.

    Parameters:
    - model: Trained Random Forest model.
    - feature_names: List of feature names.
    - machine_name: Name of the machine (e.g., "Cylinder Bottom Saw", "CNC Lathe").
    """
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="viridis")
    plt.title(f"Feature Importance - Random Forest ({machine_name})")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

# Plot for Saw and Lathe
plot_feature_importance(saw_rf, saw_X.columns, "Cylinder Bottom Saw")
plot_feature_importance(lathe_rf, lathe_X.columns, "CNC Lathe")


#-------------------------------------------

from sklearn.metrics import precision_recall_curve, auc

def plot_precision_recall(y_true, y_pred, machine_name):
    """
    Plots the Precision-Recall curve to evaluate model performance on imbalanced anomaly detection.

    Parameters:
    - y_true: Ground truth anomaly labels.
    - y_pred: Model predicted probabilities (not hard labels).
    - machine_name: Name of the machine (e.g., "Cylinder Bottom Saw", "CNC Lathe").
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label=f"AUC = {auc_score:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - Random Forest ({machine_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Compute predicted probabilities for Precision-Recall Curve
saw_rf_probs = saw_rf.predict_proba(saw_X)[:, 1]  # Get probability for anomaly class
lathe_rf_probs = lathe_rf.predict_proba(lathe_X)[:, 1]  # Get probability for anomaly class

# Plot for Saw and Lathe
plot_precision_recall(saw_y, saw_rf_probs, "Cylinder Bottom Saw")
plot_precision_recall(lathe_y, lathe_rf_probs, "CNC Lathe")







#===============================================================================

"""Feature Comparison and Selection for KNN Improvement
Feature selection is crucial to improving KNN performance since irrelevant or noisy features can distort distance calculations. We can approach this in multiple ways:

Step 1: Feature Importance Using Random Forest
Random Forest can provide feature importance scores, helping us determine which features contribute most to distinguishing anomalies.

Step 2: Correlation Analysis
Highly correlated features can introduce redundancy. We'll analyze the correlation matrix to remove unnecessary features.

Step 3: Recursive Feature Elimination (RFE)
RFE iteratively removes less important features to find an optimal feature subset."""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Extract features and target variable
features = lathe_pca_df.drop(columns=["anomaly", "timestamp"], errors="ignore")
target = lathe_pca_df["anomaly"]

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 1: Feature Importance using Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features_scaled, target)

# Extract feature importance
feature_importance = pd.DataFrame({"Feature": features.columns, "Importance": rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Step 2: Correlation Analysis
correlation_matrix = pd.DataFrame(features_scaled, columns=features.columns).corr()

# Step 3: Recursive Feature Elimination (RFE)
rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=10, step=1)
rfe_selector.fit(features_scaled, target)
selected_features = features.columns[rfe_selector.support_]

# Visualize Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Display Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Output selected features
selected_features


#===============================================================================



#===============================================================================

#===============================================================================

"""🚀 Step: Improve Anomaly Detection using One-Class SVM and Random Forest
Now that we've trained the Autoencoder, we will introduce One-Class SVM and Random Forest to improve anomaly detection.

✅ Goals of this Step:
1️⃣ Train One-Class SVM on normal reconstruction errors.
2️⃣ Train Random Forest Classifier using labeled data.
3️⃣ Compare performance of Autoencoder, One-Class SVM, and Random Forest.
4️⃣ Implement Ensemble Learning (combine models for better accuracy).

"""
# 🔹 Step 1: Train One-Class SVM on Normal Data
# One-Class SVM learns the distribution of normal data and flags outliers as anomalies.

# ✅ Train One-Class SVM on Normal Reconstruction Errors
# ==============================================================================  
from sklearn.svm import OneClassSVM

# Extract normal reconstruction errors for training
saw_train_normal = saw_pca_df[saw_pca_df["anomaly"] == 0]["reconstruction_error"].values.reshape(-1, 1)
lathe_train_normal = lathe_pca_df[lathe_pca_df["anomaly"] == 0]["reconstruction_error"].values.reshape(-1, 1)

# Train One-Class SVM
saw_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)  # nu controls anomaly detection sensitivity
lathe_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)

saw_svm.fit(saw_train_normal)
lathe_svm.fit(lathe_train_normal)

print("✅ One-Class SVM trained on normal reconstruction errors.")

#-------------------------------------------

# 🔹 Step 2: Predict Anomalies using One-Class SVM
# Now, we apply the trained One-Class SVM to detect anomalies.

# ==============================================================================  
# ✅ Predict Anomalies Using One-Class SVM
# ==============================================================================  

# Compute predictions on full dataset
saw_pca_df["svm_predicted_anomaly"] = saw_svm.predict(saw_pca_df["reconstruction_error"].values.reshape(-1, 1))
lathe_pca_df["svm_predicted_anomaly"] = lathe_svm.predict(lathe_pca_df["reconstruction_error"].values.reshape(-1, 1))

# Convert SVM predictions: -1 (outlier) → 1 (anomaly), 1 (normal) → 0 (normal)
saw_pca_df["svm_predicted_anomaly"] = saw_pca_df["svm_predicted_anomaly"].apply(lambda x: 1 if x == -1 else 0)
lathe_pca_df["svm_predicted_anomaly"] = lathe_pca_df["svm_predicted_anomaly"].apply(lambda x: 1 if x == -1 else 0)

# Evaluate Performance
from sklearn.metrics import classification_report

print("🔹 Cylinder Bottom Saw - One-Class SVM Performance")
print(classification_report(saw_pca_df["anomaly"], saw_pca_df["svm_predicted_anomaly"]))

print("\n🔹 CNC Lathe - One-Class SVM Performance")
print(classification_report(lathe_pca_df["anomaly"], lathe_pca_df["svm_predicted_anomaly"]))



#-------------------------------------------

# 2: Run Individually

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

# Set individual nu value
nu = 0.05

# Train One-Class SVM
print(f"Training One-Class SVM with nu={nu}...")
lathe_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
lathe_svm.fit(lathe_train_normal)

# Predict anomalies on full dataset
lathe_pca_df[f"svm_pred_{nu}"] = lathe_svm.predict(lathe_pca_df["reconstruction_error"].values.reshape(-1, 1))

# Convert SVM predictions: -1 (outlier) → 1 (anomaly), 1 (normal) → 0 (normal)
lathe_pca_df[f"svm_pred_{nu}"] = lathe_pca_df[f"svm_pred_{nu}"].apply(lambda x: 1 if x == -1 else 0)

# Compute classification report
report = classification_report(lathe_pca_df["anomaly"], lathe_pca_df[f"svm_pred_{nu}"], output_dict=True)

print(f"✅ Completed for nu={nu}")
report


#-------------------------------------------

# Set new nu value
nu = 0.1

# Train One-Class SVM
print(f"Training One-Class SVM with nu={nu}...")
lathe_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
lathe_svm.fit(lathe_train_normal)

# Predict anomalies on full dataset
lathe_pca_df[f"svm_pred_{nu}"] = lathe_svm.predict(lathe_pca_df["reconstruction_error"].values.reshape(-1, 1))

# Convert SVM predictions: -1 (outlier) → 1 (anomaly), 1 (normal) → 0 (normal)
lathe_pca_df[f"svm_pred_{nu}"] = lathe_pca_df[f"svm_pred_{nu}"].apply(lambda x: 1 if x == -1 else 0)

# Compute classification report
report = classification_report(lathe_pca_df["anomaly"], lathe_pca_df[f"svm_pred_{nu}"], output_dict=True)

print(f"✅ Completed for nu={nu}")
report



#-------------------------------------------

# 🚀 Step 1: Ensure Only Valid Features Are Selected
available_features_saw = set(saw_pca_df.columns)  # Get actual column names
available_features_lathe = set(lathe_pca_df.columns)  # Get actual column names

# ✅ Filter selected features based on actual columns in the DataFrame
valid_selected_features_saw = [feature for feature in selected_features_final_saw if feature in available_features_saw]
valid_selected_features_lathe = [feature for feature in selected_features_final_lathe if feature in available_features_lathe]

# ✅ Raise an error if no valid features remain (to prevent silent failure)
if not valid_selected_features_saw:
    raise ValueError("🚨 No valid selected features found in saw_pca_df! Check feature selection process.")

if not valid_selected_features_lathe:
    raise ValueError("🚨 No valid selected features found in lathe_pca_df! Check feature selection process.")

# ✅ Proceed with correlation matrix only for valid features
corr_matrix_saw = saw_pca_df[valid_selected_features_saw].corr().abs()
corr_matrix_lathe = lathe_pca_df[valid_selected_features_lathe].corr().abs()

print("✅ Feature selection fixed! Using", len(valid_selected_features_saw), "features for Saw.")
print("✅ Feature selection fixed! Using", len(valid_selected_features_lathe), "features for Lathe.")

#===============================================================================
#===============================================================================



















#==============================================================================


"""For re-running the One-Class SVM, I primarily used information from feature selection and correlation analysis:

Feature Importance (Random Forest)

Identified which features contribute the most to anomaly detection.
Ensured that reconstruction error was included, as it plays a significant role in SVM-based anomaly detection.
Dropped low-importance features that had near-zero contribution.
Feature Correlation Heatmap

Removed highly correlated features (correlation > 0.9).
Kept only independent features to avoid redundancy in the One-Class SVM training.
How This Was Applied to Re-Run SVM
Used reconstruction error (since it's a critical anomaly detection feature).
Ensured that only filtered features from the Random Forest and correlation analysis were retained before passing them to One-Class SVM.
The filtered dataset was then used for training One-Class SVM with nu=0.01 and nu=0.1



🚀 Optimized One-Class SVM & KNN with Feature Selection



"""
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy.stats import mode




# ✅ Step 1: Identify Features that Actually Exist in the DataFrame
available_features_saw = set(saw_pca_df.columns)
available_features_lathe = set(lathe_pca_df.columns)

# ✅ Step 2: Filter Only Features That Exist
valid_selected_features_saw = [feature for feature in selected_features_final_saw if feature in available_features_saw]
valid_selected_features_lathe = [feature for feature in selected_features_final_lathe if feature in available_features_lathe]

# ✅ Print the Final List of Valid Features
print("✅ Final Valid Features for Saw:", valid_selected_features_saw)
print("✅ Final Valid Features for Lathe:", valid_selected_features_lathe)


# 🚀 Step 1: Feature Selection Based on Importance & Correlation Analysis

# ✅ Step 1: Select Features for Both Machines
# Only keep features that were identified as important based on Random Forest
selected_features_final_saw = feature_importance[feature_importance["Importance"] > 0.005]["Feature"].tolist()
selected_features_final_lathe = selected_features_final_saw.copy()

# ✅ Remove Highly Correlated Features (> 0.9) to avoid redundancy
correlation_threshold = 0.9

# Saw Feature Selection
corr_matrix_saw = saw_pca_df[selected_features_final_saw].corr().abs()
upper_saw = corr_matrix_saw.where(np.triu(np.ones(corr_matrix_saw.shape), k=1).astype(bool))
to_drop_saw = [column for column in upper_saw.columns if any(upper_saw[column] > correlation_threshold)]
selected_features_final_saw = [col for col in selected_features_final_saw if col not in to_drop_saw]

# Lathe Feature Selection
corr_matrix_lathe = lathe_pca_df[selected_features_final_lathe].corr().abs()
upper_lathe = corr_matrix_lathe.where(np.triu(np.ones(corr_matrix_lathe.shape), k=1).astype(bool))
to_drop_lathe = [column for column in upper_lathe.columns if any(upper_lathe[column] > correlation_threshold)]
selected_features_final_lathe = [col for col in selected_features_final_lathe if col not in to_drop_lathe]

print("✅ Final Selected Features for Saw:", selected_features_final_saw)
print("✅ Final Selected Features for Lathe:", selected_features_final_lathe)



#==============================================================================
# 🚀 Step 2: Prepare Data Using Selected Features

#==============================================================================

# ✅ Extract Features and Labels
saw_selected_X = saw_pca_df[selected_features_final_saw]
lathe_selected_X = lathe_pca_df[selected_features_final_lathe]
saw_y = saw_pca_df["anomaly"]
lathe_y = lathe_pca_df["anomaly"]

# ✅ Split Data into Train & Test
saw_X_train, saw_X_test, saw_y_train, saw_y_test = train_test_split(saw_selected_X, saw_y, test_size=0.2, random_state=42)
lathe_X_train, lathe_X_test, lathe_y_train, lathe_y_test = train_test_split(lathe_selected_X, lathe_y, test_size=0.2, random_state=42)

#==============================================================================



# 🚀 Step 3: Train & Evaluate One-Class SVM

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ----------------------------
# 🚀 Step 1: Prepare Data with Selected Features
# ----------------------------

# ✅ Use only the finalized selected features
saw_selected_X = saw_pca_df[valid_selected_features_saw]
lathe_selected_X = lathe_pca_df[valid_selected_features_lathe]
saw_y = saw_pca_df["anomaly"]
lathe_y = lathe_pca_df["anomaly"]

# ✅ Extract only normal instances for training
saw_train_normal = saw_selected_X[saw_pca_df["anomaly"] == 0]
lathe_train_normal = lathe_selected_X[lathe_pca_df["anomaly"] == 0]

# ----------------------------
# 🚀 Step 2: Train One-Class SVM
# ----------------------------

nu = 0.1  # Adjust sensitivity

print(f"🔹 Training One-Class SVM for Cylinder Bottom Saw with nu={nu}...")
saw_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
saw_svm.fit(saw_train_normal)

print(f"🔹 Training One-Class SVM for CNC Lathe with nu={nu}...")
lathe_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
lathe_svm.fit(lathe_train_normal)

# ----------------------------
# 🚀 Step 3: Predict Anomalies
# ----------------------------

# ✅ Predict anomalies on full dataset
saw_pca_df[f"svm_pred_{nu}"] = saw_svm.predict(saw_selected_X)
lathe_pca_df[f"svm_pred_{nu}"] = lathe_svm.predict(lathe_selected_X)

# ✅ Convert Predictions: -1 (outlier) → 1 (anomaly), 1 (normal) → 0 (normal)
saw_pca_df[f"svm_pred_{nu}"] = saw_pca_df[f"svm_pred_{nu}"].apply(lambda x: 1 if x == -1 else 0)
lathe_pca_df[f"svm_pred_{nu}"] = lathe_pca_df[f"svm_pred_{nu}"].apply(lambda x: 1 if x == -1 else 0)

print("✅ One-Class SVM Predictions Completed")

# ----------------------------
# 🚀 Step 4: Evaluate Performance (Saw & Lathe)
# ----------------------------

def evaluate_svm(y_true, y_pred, machine_name):
    print(f"\n🔹 {machine_name} - One-Class SVM (nu={nu}) Performance")
    print(classification_report(y_true, y_pred))

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - One-Class SVM ({machine_name})")
    plt.show()

    # Compute ROC Curve & AUC Score
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray")  # Random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - One-Class SVM ({machine_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

# ✅ Evaluate Saw & Lathe
evaluate_svm(saw_y, saw_pca_df[f"svm_pred_{nu}"], "Cylinder Bottom Saw")
evaluate_svm(lathe_y, lathe_pca_df[f"svm_pred_{nu}"], "CNC Lathe")

print("🚀 One-Class SVM Evaluation Completed for Both Machines")




#==============================================================================

# 🚀 Step 4: Train & Evaluate KNN Classifie

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# ----------------------------
# 🚀 Step 1: Prepare Data for KNN
# ----------------------------

# ✅ Extract Features and Labels
saw_selected_X = saw_pca_df[valid_selected_features_saw]
lathe_selected_X = lathe_pca_df[valid_selected_features_lathe]
saw_y = saw_pca_df["anomaly"]
lathe_y = lathe_pca_df["anomaly"]

# ✅ Split Data into Train & Test
saw_X_train, saw_X_test, saw_y_train, saw_y_test = train_test_split(saw_selected_X, saw_y, test_size=0.2, random_state=42)
lathe_X_train, lathe_X_test, lathe_y_train, lathe_y_test = train_test_split(lathe_selected_X, lathe_y, test_size=0.2, random_state=42)

# ----------------------------
# 🚀 Step 2: Train KNN with Different Distance Metrics
# ----------------------------

# Define different distance metrics
distance_metrics = ["euclidean", "manhattan", "minkowski", "cosine"]

# Store results
knn_results = {}

# Loop through distance metrics and evaluate KNN
for metric in distance_metrics:
    print(f"🔹 Training KNN for Saw & Lathe with {metric} distance...")

    # Train KNN for Saw
    knn_saw = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn_saw.fit(saw_X_train, saw_y_train)
    saw_pred = knn_saw.predict(saw_X_test)

    # Train KNN for Lathe
    knn_lathe = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn_lathe.fit(lathe_X_train, lathe_y_train)
    lathe_pred = knn_lathe.predict(lathe_X_test)

    # Store Predictions
    saw_pca_df[f"knn_pred_{metric}"] = knn_saw.predict(saw_selected_X)
    lathe_pca_df[f"knn_pred_{metric}"] = knn_lathe.predict(lathe_selected_X)

    # Evaluate Saw & Lathe
    print(f"\n🔹 {metric.upper()} Distance - Cylinder Bottom Saw Performance")
    print(classification_report(saw_y_test, saw_pred))

    print(f"\n🔹 {metric.upper()} Distance - CNC Lathe Performance")
    print(classification_report(lathe_y_test, lathe_pred))

    # Store results
    knn_results[metric] = {
        "saw_confusion_matrix": confusion_matrix(saw_y_test, saw_pred),
        "lathe_confusion_matrix": confusion_matrix(lathe_y_test, lathe_pred)
    }

    # Plot Confusion Matrices
    plt.figure(figsize=(6,5))
    sns.heatmap(knn_results[metric]["saw_confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - KNN ({metric}) - Cylinder Bottom Saw")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(knn_results[metric]["lathe_confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - KNN ({metric}) - CNC Lathe")
    plt.show()

print("🚀 KNN Evaluation Completed for Both Machines")

# ✅ Select Best KNN Predictions for Ensemble
saw_pca_df["knn_pred"] = saw_pca_df["knn_pred_euclidean"]  # Any (since results are similar)
lathe_pca_df["knn_pred"] = lathe_pca_df["knn_pred_manhattan"]  # Best-performing metric for Lathe

print("✅ Best KNN models selected for ensemble voting.")

#==============================================================================

# # 🚀 Step 5: Ensemble Learning Using Majority Voting

# # ✅ Aggregate Predictions for Ensemble Model
# saw_pca_df["ensemble_predicted_anomaly"] = mode(
#     saw_pca_df[["optimized_predicted_anomaly", "rf_predicted_anomaly", f"svm_pred_{nu}", "knn_pred_euclidean"]],
#     axis=1
# )[0].flatten()

# lathe_pca_df["ensemble_predicted_anomaly"] = mode(
#     lathe_pca_df[["optimized_predicted_anomaly", "rf_predicted_anomaly", f"svm_pred_{nu}", "knn_pred_euclidean"]],
#     axis=1
# )[0].flatten()

# print("✅ Ensemble Model Applied Successfully to Both Machines")


#==============================================================================

# 🚀 Step: Ensemble Voting for Anomaly Detection
# Now, we will integrate Autoencoder, Random Forest, One-Class SVM, and the best KNN models into an ensemble model using majority voting.


"""🔹 Why Use an Ensemble?
✅ Reduces overfitting – Each model has different strengths; ensemble voting balances them.
✅ Boosts robustness – Outliers affecting one model might not affect all.
✅ Improves generalization – Aggregating diverse methods leads to more accurate anomaly detection.
✅ More reliable results – If different models agree, it provides a stronger anomaly signal."""

#==============================================================================

# ✅ Step 1: Aggregate Predictions from Each Model

from scipy.stats import mode
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ✅ For Cylinder Bottom Saw
saw_pca_df["autoencoder_pred"] = saw_pca_df["optimized_predicted_anomaly"]  # Autoencoder
saw_pca_df["random_forest_pred"] = saw_pca_df["rf_predicted_anomaly"]  # Random Forest
saw_pca_df["svm_pred"] = saw_pca_df["svm_pred_0.1"]  # One-Class SVM
saw_pca_df["knn_pred"] = saw_pca_df["knn_pred"]  # Best KNN (Euclidean)

# ✅ For CNC Lathe
lathe_pca_df["autoencoder_pred"] = lathe_pca_df["optimized_predicted_anomaly"]  # Autoencoder
lathe_pca_df["random_forest_pred"] = lathe_pca_df["rf_predicted_anomaly"]  # Random Forest
lathe_pca_df["svm_pred"] = lathe_pca_df["svm_pred_0.1"]  # One-Class SVM
lathe_pca_df["knn_pred"] = lathe_pca_df["knn_pred"]  # Best KNN (Cosine)

# ✅ Apply Majority Voting
saw_pca_df["ensemble_predicted_anomaly"] = mode(
    saw_pca_df[["autoencoder_pred", "random_forest_pred", "svm_pred", "knn_pred"]], axis=1
)[0].flatten()

lathe_pca_df["ensemble_predicted_anomaly"] = mode(
    lathe_pca_df[["autoencoder_pred", "random_forest_pred", "svm_pred", "knn_pred"]], axis=1
)[0].flatten()

print("✅ Majority Voting Applied Successfully to Both Saw & Lathe")

#-------------------------------------------

# ✅ Step 2: Evaluate Ensemble Performance


# Function to evaluate ensemble performance
def evaluate_ensemble(y_true, y_pred, machine_name):
    print(f"\n🔹 {machine_name} - Ensemble Model Performance")
    print(classification_report(y_true, y_pred))

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Ensemble ({machine_name})")
    plt.show()

# ✅ Evaluate Ensemble Model for Saw & Lathe
evaluate_ensemble(saw_pca_df["anomaly"], saw_pca_df["ensemble_predicted_anomaly"], "Cylinder Bottom Saw")
evaluate_ensemble(lathe_pca_df["anomaly"], lathe_pca_df["ensemble_predicted_anomaly"], "CNC Lathe")
#-------------------------------------------

# ✅ Step 3: ROC Curve for Ensemble Model


# Function to plot ROC Curve
def plot_roc_curve(y_true, y_pred, machine_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray")  # Random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Ensemble Model ({machine_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

# ✅ Plot ROC Curve for Saw & Lathe
plot_roc_curve(saw_pca_df["anomaly"], saw_pca_df["ensemble_predicted_anomaly"], "Cylinder Bottom Saw")
plot_roc_curve(lathe_pca_df["anomaly"], lathe_pca_df["ensemble_predicted_anomaly"], "CNC Lathe")

#-------------------------------------------

# Function to compare performance of all models
def compare_models(df, machine_name):
    results = {}
    model_names = ["autoencoder_pred", "random_forest_pred", "svm_pred", "knn_pred", "ensemble_predicted_anomaly"]

    for model in model_names:
        report = classification_report(df["anomaly"], df[model], output_dict=True)
        results[model] = {
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-Score": report["1"]["f1-score"]
        }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    print(f"\n🔹 Performance Comparison - {machine_name}:")
    print(results_df)

# ✅ Compare models for Saw & Lathe
compare_models(saw_pca_df, "Cylinder Bottom Saw")
compare_models(lathe_pca_df, "CNC Lathe")

print("🚀 Ensemble Learning Completed for Both Machines")

#-------------------------------------------








#===============================================================================

"""🔹 Step 5: Combine Models using an Ensemble Approach
We now create an Ensemble Model that takes predictions from:

Autoencoder
One-Class SVM
Random Forest
We apply a majority vote. """
 
# ✅ Ensemble Model: Majority Voting
# ==============================================================================  
def majority_vote(df, model_columns):
    """
    Performs majority voting across multiple anomaly detection models.

    Args:
        df (DataFrame): Data containing anomaly predictions.
        model_columns (list): List of model prediction columns.

    Returns:
        pd.Series: Final ensemble predictions.
    """
    return df[model_columns].mode(axis=1)[0]

# Apply majority voting
saw_pca_df["ensemble_predicted_anomaly"] = majority_vote(saw_pca_df, ["predicted_anomaly", "svm_predicted_anomaly", "rf_predicted_anomaly"])
lathe_pca_df["ensemble_predicted_anomaly"] = majority_vote(lathe_pca_df, ["predicted_anomaly", "svm_predicted_anomaly", "rf_predicted_anomaly"])

# Evaluate Performance
print("🔹 Cylinder Bottom Saw - Ensemble Model Performance")
print(classification_report(saw_y, saw_pca_df["ensemble_predicted_anomaly"]))

print("\n🔹 CNC Lathe - Ensemble Model Performance")
print(classification_report(lathe_y, lathe_pca_df["ensemble_predicted_anomaly"]))


# ==============================================================================  

# 🔹 Step 6: Visualizing Anomaly Predictions
# ==============================================================================  
import seaborn as sns

def plot_anomaly_comparison(df, title):
    """
    Plots actual vs. predicted anomalies.
    
    Args:
        df (DataFrame): Data containing actual and predicted anomalies.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 5))
    sns.histplot(df["anomaly"], label="Actual Anomalies", color="blue", kde=True, alpha=0.5)
    sns.histplot(df["ensemble_predicted_anomaly"], label="Predicted Anomalies", color="red", kde=True, alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.show()

# Plot anomaly detection performance
plot_anomaly_comparison(saw_pca_df, "Cylinder Bottom Saw - Anomaly Detection Performance")
plot_anomaly_comparison(lathe_pca_df, "CNC Lathe - Anomaly Detection Performance")

# ==============================================================================



from sklearn.ensemble import IsolationForest

# Train Isolation Forest Model
iso_forest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
lathe_pca_df["if_pred"] = iso_forest.fit_predict(lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error"], errors="ignore"))

# Convert predictions (-1 = anomaly, 1 = normal) to (1 = anomaly, 0 = normal)
lathe_pca_df["if_pred"] = lathe_pca_df["if_pred"].apply(lambda x: 1 if x == -1 else 0)

# Evaluate Performance
print("🔹 CNC Lathe - Isolation Forest Performance")
print(classification_report(lathe_pca_df["anomaly"], lathe_pca_df["if_pred"]))
#===============================================================================

from sklearn.neighbors import NearestNeighbors

# Train KNN Model
knn = NearestNeighbors(n_neighbors=5)
knn.fit(lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error"], errors="ignore"))

# Compute distances to nearest neighbors
distances, _ = knn.kneighbors(lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error"], errors="ignore"))

# Set anomaly threshold based on 95th percentile distance
threshold = np.percentile(distances, 95)
lathe_pca_df["knn_pred"] = (distances.mean(axis=1) > threshold).astype(int)

# Evaluate Performance
print("🔹 CNC Lathe - KNN Performance")
print(classification_report(lathe_pca_df["anomaly"], lathe_pca_df["knn_pred"]))
#================================================================

from sklearn.mixture import GaussianMixture

# Train Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
lathe_pca_df["gmm_score"] = gmm.fit(lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error"], errors="ignore")).score_samples(lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error"], errors="ignore"))

# Convert anomaly scores to binary labels (lower scores = more anomalous)
threshold = lathe_pca_df["gmm_score"].quantile(0.05)  # Set threshold at 5th percentile
lathe_pca_df["gmm_pred"] = (lathe_pca_df["gmm_score"] < threshold).astype(int)

# Evaluate Performance
print("🔹 CNC Lathe - Gaussian Mixture Model (GMM) Performance")
print(classification_report(lathe_pca_df["anomaly"], lathe_pca_df["gmm_pred"]))
#===============================================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Prepare feature matrix (X) and target labels (y)
lathe_X = lathe_pca_df.drop(columns=["anomaly", "timestamp", "reconstruction_error"], errors="ignore")
lathe_y = lathe_pca_df["anomaly"]

# Split dataset into training (80%) and testing (20%)
lathe_X_train, lathe_X_test, lathe_y_train, lathe_y_test = train_test_split(
    lathe_X, lathe_y, test_size=0.2, random_state=42
)

print("✅ Train-Test Split Completed!")
print(f"Training samples: {lathe_X_train.shape[0]}, Testing samples: {lathe_X_test.shape[0]}")

# Define parameter grid
param_grid = {'n_neighbors': [3, 5, 10, 15, 20]}

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Perform Grid Search to find best k
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="f1_macro")
grid_search.fit(lathe_X_train, lathe_y_train)

# Print best k value
print(f"✅ Best k: {grid_search.best_params_['n_neighbors']}")

# Train KNN with best k
best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
best_knn.fit(lathe_X_train, lathe_y_train)




"""Now that we have k=3, let’s:

1️⃣ Train KNN with k=3
2️⃣ Evaluate performance (Precision, Recall, F1-score)
3️⃣ Visualize results (Confusion Matrix, ROC Curve, etc.)"""


#-------------------------------------------

# ✅ Step 1: Train the Final KNN Model
best_knn = KNeighborsClassifier(n_neighbors=3)
best_knn.fit(lathe_X_train, lathe_y_train)

# Predict on test data
lathe_knn_predictions = best_knn.predict(lathe_X_test)

# Evaluate performance
from sklearn.metrics import classification_report, confusion_matrix

print("🔹 CNC Lathe - KNN (k=3) Performance")
print(classification_report(lathe_y_test, lathe_knn_predictions))
#-------------------------------------------

# ✅ Step 2: Visualize Confusion Matrix
cm = confusion_matrix(lathe_y_test, lathe_knn_predictions)

# Plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN (k=3)")
plt.show()
#-------------------------------------------

# ✅ Step 3: Plot ROC Curve & AUC Score
from sklearn.metrics import roc_curve, auc

# Get prediction probabilities
lathe_knn_probs = best_knn.predict_proba(lathe_X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(lathe_y_test, lathe_knn_probs)
auc_score = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray")  # Random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN (k=3)")
plt.legend()
plt.grid(True)
plt.show()
#-------------------------------------------



"""🔍 Experiment Results for Different Distance Metrics in KNN
I will generate:

Confusion Matrices: To visualize the number of correct/incorrect classifications.
ROC Curves & AUC Scores: To measure the overall classification performance.
Classification Reports: Showing precision, recall, and F1-score."""

# 📊 Step 1: Confusion Matrices

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Define the distance metrics
distance_metrics = ["euclidean", "manhattan", "minkowski", "cosine"]

# Store results
knn_results = {}

# Loop through each distance metric and evaluate KNN
for metric in distance_metrics:
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn.fit(lathe_X_train, lathe_y_train)
    
    # Predict anomalies on test data
    y_pred = knn.predict(lathe_X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(lathe_y_test, y_pred)
    
    # Compute ROC curve & AUC score
    fpr, tpr, _ = roc_curve(lathe_y_test, y_pred)
    auc_score = auc(fpr, tpr)
    
    # Store results
    knn_results[metric] = {
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr, auc_score),
        "classification_report": classification_report(lathe_y_test, y_pred, output_dict=True)
    }

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - KNN ({metric} distance)")
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray")  # Random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - KNN ({metric} distance)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Display classification reports
knn_results

#===============================================================================

"""Feature Selection:

We can remove less important features (those with near-zero importance).
Drop highly correlated features (one of each pair with correlation >0.9).
Retain the top PCA components and key anomaly detection outputs.
Ensemble Learning (Hybrid Model)

Now that we know the feature importances, we can combine the best models (Autoencoder, KNN, SVM, GMM, Isolation Forest) using an ensemble strategy."""
import numpy as np

# Define correlation threshold for feature selection
correlation_threshold = 0.9

# Compute correlation matrix
corr_matrix = feature_df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than threshold
to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

# Drop highly correlated features
filtered_feature_df = feature_df.drop(columns=to_drop)

# Drop low-importance features (threshold set to 0.005 for filtering)
low_importance_threshold = 0.005
important_features = rf_feature_importance[rf_feature_importance['Importance Score'] > low_importance_threshold]['Features'].values

# Select only the important features
final_selected_features = [f for f in filtered_feature_df.columns if f in important_features]

# Create final dataset with selected features
final_feature_df = filtered_feature_df[final_selected_features]

# Display the filtered feature list
final_feature_df.head()

#-------------------------------------------

import numpy as np
import pandas as pd

# Simulate feature dataframe (since execution state was reset)
# Replace with actual feature data
np.random.seed(42)
feature_names = [f"PC{i}" for i in range(1, 51)] + ["reconstruction_error", "adjusted_predicted_anomaly", "svm_predicted_anomaly", "svm_pred_0.1", "knn_pred", "gmm_pred"]
feature_df = pd.DataFrame(np.random.rand(1000, len(feature_names)), columns=feature_names)

# Define correlation threshold for feature selection
correlation_threshold = 0.9

# Compute correlation matrix
corr_matrix = feature_df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than threshold
to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

# Drop highly correlated features
filtered_feature_df = feature_df.drop(columns=to_drop)

# Simulate feature importance using Random Forest (since execution was reset)
rf_feature_importance = pd.DataFrame({
    "Features": filtered_feature_df.columns,
    "Importance Score": np.random.rand(len(filtered_feature_df.columns))
}).sort_values(by="Importance Score", ascending=False)

# Drop low-importance features (threshold set to 0.005 for filtering)
low_importance_threshold = 0.005
important_features = rf_feature_importance[rf_feature_importance['Importance Score'] > low_importance_threshold]['Features'].values

# Select only the important features
final_selected_features = [f for f in filtered_feature_df.columns if f in important_features]

# Create final dataset with selected features
final_feature_df = filtered_feature_df[final_selected_features]

# Display the filtered feature list

# Extract the top 20 important features
top_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(20)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(y=top_features["Feature"], x=top_features["Importance"], palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 20 Important Features (Filtered)")
plt.show()



# Ensure the feature importance data exists before plotting
if "Feature" in feature_importance_df.columns and "Importance" in feature_importance_df.columns:
    # Extract the top 20 important features again
    top_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(20)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(y=top_features["Feature"], x=top_features["Importance"], palette="viridis")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Top 20 Important Features (Filtered)")
    plt.show()
else:
    print("Feature importance data is missing or not correctly formatted.")









#===============================================================================
#================================================================================

import os  # File paths
import logging  # Logging for debugging
import numpy as np
import pandas as pd
import joblib  # Model persistence
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_curve, roc_curve, auc, precision_recall_fscore_support
)
from sklearn.feature_selection import RFE
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import IsolationForest

# =============================================
# 🔹 Step 1: Data Loading & Inspection
# =============================================

def load_sensor_data(file_path):
    """Loads a processed CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# Load data files
saw_df = load_sensor_data("path_to_saw_data.csv")
lathe_df = load_sensor_data("path_to_lathe_data.csv")

# Convert timestamp to datetime
saw_df['timestamp'] = pd.to_datetime(saw_df['timestamp'])
lathe_df['timestamp'] = pd.to_datetime(lathe_df['timestamp'])

# =============================================
# 🔹 Step 2: Data Preprocessing
# =============================================

def handle_missing_values(df):
    """Handles missing values."""
    df = df.dropna(thresh=len(df) * 0.5, axis=1)  # Drop columns with >50% missing
    for column in df.columns:
        df[column] = df[column].fillna(df[column].median())  # Fill missing with median
    return df

saw_df = handle_missing_values(saw_df)
lathe_df = handle_missing_values(lathe_df)

# Feature scaling (Standardization)
def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

saw_df, saw_scaler = scale_features(saw_df)
lathe_df, lathe_scaler = scale_features(lathe_df)

# =============================================
# 🔹 Step 3: Feature Selection
# =============================================

def remove_correlated_features(df, threshold=0.9):
    """Removes highly correlated features."""
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return df.drop(columns=to_drop)

saw_df = remove_correlated_features(saw_df)
lathe_df = remove_correlated_features(lathe_df)

# =============================================
# 🔹 Step 4: Apply PCA
# =============================================

def apply_pca(df, variance_threshold=0.95):
    """Applies PCA for dimensionality reduction."""
    normal_data = df[df['anomaly'] == 0].drop(columns=['anomaly', 'timestamp'])
    scaler = StandardScaler()
    normal_data_scaled = scaler.fit_transform(normal_data)
    pca = PCA(n_components=variance_threshold)
    pca.fit(normal_data_scaled)
    return pca, scaler

saw_pca, saw_scaler = apply_pca(saw_df)
lathe_pca, lathe_scaler = apply_pca(lathe_df)

# =============================================
# 🔹 Step 5: Train Anomaly Detection Models
# =============================================

def train_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(input_layer)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(decoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

saw_autoencoder = train_autoencoder(saw_df.shape[1])
lathe_autoencoder = train_autoencoder(lathe_df.shape[1])

# Train One-Class SVM
saw_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
lathe_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)

saw_svm.fit(saw_df.drop(columns=['anomaly', 'timestamp']))
lathe_svm.fit(lathe_df.drop(columns=['anomaly', 'timestamp']))

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, random_state=42)
iso_forest.fit(lathe_df.drop(columns=['anomaly', 'timestamp']))

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(lathe_df.drop(columns=['anomaly', 'timestamp']), lathe_df['anomaly'])

# Train GMM
gmm = GaussianMixture(n_components=2)
gmm.fit(lathe_df.drop(columns=['anomaly', 'timestamp']))

# =============================================
# 🔹 Step 6: Model Optimization & Tuning
# =============================================

# Grid Search for best K in KNN
param_grid = {'n_neighbors': [3, 5, 10, 15, 20]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="f1_macro")
grid_search.fit(lathe_df.drop(columns=['anomaly', 'timestamp']), lathe_df['anomaly'])
best_k = grid_search.best_params_['n_neighbors']

# =============================================
# 🔹 Step 7: Ensemble Learning
# =============================================

def majority_vote(df, model_columns):
    return df[model_columns].mode(axis=1)[0]

# Apply majority voting
lathe_df['ensemble_predicted_anomaly'] = majority_vote(lathe_df, ['svm_predicted_anomaly', 'rf_predicted_anomaly'])

# =============================================
# 🔹 Step 8: Performance Evaluation & Visualization
# =============================================

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
    plt.title(title)
    plt.show()

plot_confusion_matrix(lathe_df['anomaly'], lathe_df['ensemble_predicted_anomaly'], "Ensemble Model Confusion Matrix")











#===============================================================================

# Extra visualization for the report

#===============================================================================

# 1. 3D Visualization of Ensemble Predictions

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_ensemble(df, title):
    """
    Plots a 3D scatter plot of the first three principal components with ensemble predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing PCA components and ensemble predictions.
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot normal points
    normal_points = df[df["ensemble_predicted_anomaly"] == 0]
    ax.scatter(normal_points["PC1"], normal_points["PC2"], normal_points["PC3"], 
               c="blue", label="Normal", alpha=0.6, s=20)
    
    # Plot anomalous points
    anomaly_points = df[df["ensemble_predicted_anomaly"] == 1]
    ax.scatter(anomaly_points["PC1"], anomaly_points["PC2"], anomaly_points["PC3"], 
               c="red", label="Anomaly", alpha=0.6, s=20)
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()
    plt.show()

# Call the function for Saw and Lathe
plot_3d_ensemble(saw_pca_df, "Cylinder Bottom Saw - Ensemble Predictions (3D)")
plot_3d_ensemble(lathe_pca_df, "CNC Lathe - Ensemble Predictions (3D)")


#+==============================================================================

#  Precision-Recall Surface Plot

from sklearn.metrics import precision_recall_curve

def plot_precision_recall_surface(df, title):
    """
    Plots a 3D surface of Precision, Recall, and F1-Score across thresholds.
    
    Args:
        df (pd.DataFrame): DataFrame containing anomaly labels and ensemble predictions.
        title (str): Title of the plot.
    """
    thresholds = np.linspace(0, 1, 100)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (df["ensemble_predicted_anomaly"] > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(df["anomaly"], y_pred, average="binary")
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(recall_scores, precision_scores, f1_scores, cmap="viridis", alpha=0.8)
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_zlabel("F1-Score")
    ax.set_title(title)
    plt.show()

# Call the function for Saw and Lathe
plot_precision_recall_surface(saw_pca_df, "Cylinder Bottom Saw - Precision-Recall-F1 Surface")
plot_precision_recall_surface(lathe_pca_df, "CNC Lathe - Precision-Recall-F1 Surface")



def plot_roc_curve(y_true, y_pred, title):
    """
    Plots the ROC curve with AUC score.
    
    Args:
        y_true (pd.Series): Ground truth labels.
        y_pred (pd.Series): Predicted labels.
        title (str): Title of the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray")  # Random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function for Saw and Lathe
plot_roc_curve(saw_pca_df["anomaly"], saw_pca_df["ensemble_predicted_anomaly"], 
               "Cylinder Bottom Saw - ROC Curve")
plot_roc_curve(lathe_pca_df["anomaly"], lathe_pca_df["ensemble_predicted_anomaly"], 
               "CNC Lathe - ROC Curve")





def plot_feature_importance(importance, feature_names, title):
    """
    Plots feature importance as a bar plot.
    
    Args:
        importance (np.array): Feature importance scores.
        feature_names (list): List of feature names.
        title (str): Title of the plot.
    """
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="viridis")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title(title)
    plt.show()

# Example: Plot feature importance for Random Forest in the ensemble
plot_feature_importance(saw_rf.feature_importances_, saw_X.columns, 
                        "Cylinder Bottom Saw - Feature Importance")
plot_feature_importance(lathe_rf.feature_importances_, lathe_X.columns, 
                        "CNC Lathe - Feature Importance")



def plot_time_series_with_anomalies(df, sensor_column, title):
    """
    Plots time-series data with anomalies highlighted.
    
    Args:
        df (pd.DataFrame): DataFrame containing sensor data and anomaly labels.
        sensor_column (str): Name of the sensor column to plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df[sensor_column], label="Sensor Reading", alpha=0.7)
    anomalies = df[df["ensemble_predicted_anomaly"] == 1]
    plt.scatter(anomalies["timestamp"], anomalies[sensor_column], color="red", label="Anomalies", zorder=3)
    plt.xlabel("Time")
    plt.ylabel(sensor_column)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function for Saw and Lathe
plot_time_series_with_anomalies(saw_pca_df, "SpindleRPM", "Cylinder Bottom Saw - Anomalies Over Time")
plot_time_series_with_anomalies(lathe_pca_df, "SpindleRPM", "CNC Lathe - Anomalies Over Time")



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Assuming lathe_pca_df contains the actual PCA-transformed data and ensemble predictions
# Selecting three principal components for visualization
pca_columns = ["PC1", "PC2", "PC3"]

# Check if required columns exist
if all(col in lathe_pca_df.columns for col in pca_columns + ["ensemble_predicted_anomaly"]):
    # Extract relevant data
    ensemble_df = lathe_pca_df[pca_columns + ["ensemble_predicted_anomaly"]]

    # Create a 3D scatter plot using actual PCA-transformed data
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Separate normal and anomaly points based on the ensemble model prediction
    normal = ensemble_df[ensemble_df["ensemble_predicted_anomaly"] == 0]
    anomalies = ensemble_df[ensemble_df["ensemble_predicted_anomaly"] == 1]

    # Scatter plot with different colors for normal and anomalies
    ax.scatter(normal["PC1"], normal["PC2"], normal["PC3"], c='blue', label="Normal", alpha=0.5)
    ax.scatter(anomalies["PC1"], anomalies["PC2"], anomalies["PC3"], c='red', label="Anomalies", alpha=0.8)

    # Labels and title
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("3D Visualization of Ensemble Model Predictions")
    ax.legend()

    # Show plot
    plt.show()
else:
    print("Error: Required PCA columns or 'ensemble_predicted_anomaly' not found in the dataset.")
