import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from scipy.fftpack import fft
import joblib
from numba import jit
from multiprocessing import Pool, cpu_count

# ============================================================
# 🚀 Define Paths for Input and Output Directories
# ============================================================
INPUT_DIR = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\interim\cleaned_dataset"
OUTPUT_DIR = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\Feature_extraction"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# ============================================================
# 📥 Load All Processed Datasets from Interim Storage
# ============================================================
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
datasets = {f.split(".")[0]: pd.read_csv(os.path.join(INPUT_DIR, f)) for f in files}


# ============================================================
# 📊 Feature Extraction Strategy Overview
# ============================================================
def display_feature_extraction_strategy():
    """
    Display an overview of the feature extraction strategy per machine type.
    """
    feature_summary = {
        "Machine Type": ["CNC Milling", "CNC Lathe", "Sawing"],
        "Key Monitored Parameters": [
            "Spindle Speed, Cutting Force, Temperature, Vibration",
            "Spindle Speed, Feed Rate, Torque, Tool Position",
            "Blade Speed, Torque, Motor Load, Surface Roughness"
        ],
        "Purpose of Feature Extraction": [
            "Detect tool wear, optimize spindle efficiency",
            "Identify excessive stress, tool breakage",
            "Monitor blade condition, detect sudden resistance"
        ],
        "Extracted Features": [
            ["Mean", "Std", "Min", "Max", "Skewness", "Kurtosis", "Rate of Change", "Acceleration"],
            ["Mean", "Std", "Min", "Max", "Skewness", "Kurtosis", "Load Variability"],
            ["Mean", "Std", "Min", "Max", "Skewness", "Kurtosis", "Motor Spikes"]
        ]
    }

    feature_summary_df = pd.DataFrame(feature_summary)
    print("\n📊 Feature Extraction Strategy Overview:\n")
    print(feature_summary_df.to_string(index=False))


# Call the overview function before feature extraction
display_feature_extraction_strategy()


# ============================================================
# 🔧 Optimized Statistical Functions using `numba`
# ============================================================
@jit(nopython=True)
def fast_skew(arr):
    """Optimized skewness calculation using numba for speedup."""
    return skew(arr)

@jit(nopython=True)
def fast_kurtosis(arr):
    """Optimized kurtosis calculation using numba for speedup."""
    return kurtosis(arr)


# ============================================================
# 🔧 Optimized Feature Extraction Function
# ============================================================
def extract_features(df, machine_type, window_size=20):
    """
    Optimized Feature Extraction with faster rolling computations and numba acceleration.

    Parameters:
        df (pd.DataFrame): The dataset to extract features from.
        machine_type (str): Type of machine (e.g., 'Milling', 'Lathe', 'Sawing').
        window_size (int): Rolling window size for computing statistics.

    Returns:
        pd.DataFrame: A DataFrame containing extracted features.
    """
    features = {}

    # Iterate over all numeric columns except timestamp and part_id
    for col in df.select_dtypes(include=["number"]).columns:
        if col not in ["timestamp", "part_id"]:
            features[f"{col}_mean"] = df[col].rolling(window=window_size).mean()
            features[f"{col}_std"] = df[col].rolling(window=window_size).std()
            features[f"{col}_min"] = df[col].rolling(window=window_size).min()
            features[f"{col}_max"] = df[col].rolling(window=window_size).max()
            
            # ✅ FIX: Apply optimized `numba` functions
            features[f"{col}_skew"] = df[col].rolling(window=window_size).apply(lambda x: fast_skew(x.values), raw=True)
            features[f"{col}_kurtosis"] = df[col].rolling(window=window_size).apply(lambda x: fast_kurtosis(x.values), raw=True)

            # Additional features based on machine type
            if machine_type == "Milling":
                features[f"{col}_rate_of_change"] = df[col].diff().rolling(window=window_size).mean()
                features[f"{col}_acceleration"] = df[col].diff().diff().rolling(window=window_size).mean()
            elif machine_type == "Lathe":
                features[f"{col}_load_variability"] = df[col].rolling(window=window_size).var()
            elif machine_type == "Sawing":
                features[f"{col}_motor_spikes"] = df[col].diff().abs().rolling(window=window_size).max()

    return pd.DataFrame(features).dropna()  # Drop rows with NaNs from rolling calculations


# ============================================================
# 🚀 Parallel Processing for Feature Extraction
# ============================================================
def process_dataset(name_df):
    """Parallel function to extract features from a dataset."""
    name, df = name_df
    print(f"📊 Processing {name}...")

    if df.empty:
        print(f"⚠️ Warning: Skipping {name} because the dataset is empty.")
        return None

    # Determine machine type
    machine_type = "Milling" if "Milling" in name else "Lathe" if "Lathe" in name else "Sawing"

    # Apply forward fill before feature extraction
    df.ffill(inplace=True)
    feature_data = extract_features(df, machine_type)

    # Save extracted features in multiple formats
    feature_data.to_csv(os.path.join(OUTPUT_DIR, f"{name}_features.csv"), index=False)
    feature_data.to_pickle(os.path.join(OUTPUT_DIR, f"{name}_features.pkl"))
    feature_data.to_parquet(os.path.join(OUTPUT_DIR, f"{name}_features.parquet"))

    print(f"✅ Extracted features for {name} and saved.")
    return name, feature_data


# ============================================================
# 🚀 Execute Feature Extraction in Parallel
# ============================================================
if __name__ == "__main__":
    with Pool(cpu_count() - 1) as pool:
        results = pool.map(process_dataset, datasets.items())

    feature_datasets = {name: df for name, df in results if df is not None}

print("🚀 Feature extraction successfully completed!")

# ============================================================
# 🚀 Anomaly Detection Using Isolation Forest
# ============================================================
def detect_anomalies(df, contamination=0.01):
    """
    Detect anomalies using Isolation Forest on scaled numerical features.
    """
    numeric_df = df.select_dtypes(include=["number"])
    
    if numeric_df.empty:
        raise ValueError("❌ Error: No numeric columns found for anomaly detection!")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)
    
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    anomalies = model.fit_predict(df_scaled)
    
    return anomalies, model, scaler

# ============================================================
# 🚀 Train Isolation Forest on Each Feature-Extracted Dataset
# ============================================================
for name, df in feature_datasets.items():
    print(f"🔍 Training Isolation Forest for {name}...")

    if df.empty:
        print(f"⚠️ Skipping {name} - Dataset is empty after feature extraction.")
        continue

    anomalies, model, scaler = detect_anomalies(df)
    df["anomaly"] = anomalies  # Append anomaly scores

    df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_anomalies.csv"), index=False)
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{name}_isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{name}_scaler.pkl"))
    
    print(f"✅ Anomaly detection completed for {name}.")

print("🚀 Feature extraction and anomaly detection successfully completed!")
# ============================================================
