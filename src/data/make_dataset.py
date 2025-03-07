# ============================================================
# 🚀 Unified EDA Pipeline for Predictive Maintenance Data
# ============================================================
# 🔹 Purpose: Perform Exploratory Data Analysis (EDA) on structured machine datasets
# 🔹 Features:
#    - Reads CSV and HDF5 (.h5) files in a modular way
#    - Performs data inspection, missing value analysis, outlier detection
#    - Generates detailed Sweetviz reports
#    - Handles missing files gracefully
# ============================================================
# 📌 Import Required Libraries
import os
import h5py  # To handle large numerical datasets efficiently
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob  # For file path handling
import sweetviz as sv  # Used for automated EDA reports
from ydata_profiling import ProfileReport  # Pandas Profiling for detailed reports
from scipy.stats import zscore # For Z-score based outlier detection
import logging

# ============================================================
# 1️⃣ Define Paths & Directory Structure
# ============================================================
DATA_ROOT = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\raw"
INTERIM_DIR = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\interim"
REPORTS_DIR = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\eda_reports"
# Ensure output directories exist
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# Define dataset paths
datasets = {
    "Cylinder Quality Data": os.path.join(DATA_ROOT, "cylinder", "assembly", "quality_data", "quality_data.csv"),
    "CNC Milling Quality Data": os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "quality_data", "quality_data.csv"),
    "Saw Quality Data": os.path.join(DATA_ROOT, "cylinder_bottom", "saw", "quality_data", "quality_data.csv"),
    "Piston Rod Quality Data": os.path.join(DATA_ROOT, "piston_rod", "cnc_lathe", "quality_data", "quality_data.csv"),
}
# Debug File Path Before Loading
# Before loading, print the absolute paths to verify:
for name, path in datasets.items():
    abs_path = os.path.abspath(path)
    print(f"🔍 Checking: {abs_path}")
    
    if os.path.exists(abs_path):
        print(f"✅ {name} - File exists!")
    else:
        print(f"❌ {name} - File NOT found!")
#-------------------------------------------------------------
# 📌 HDF5 File Inspection & Metadata Extraction Utility
"""cylinder_bottom
│   ├── cnc_milling_machine
│   │   ├── process_data
│   │   │   ├── 100101_11_29_2022_12_30_29  <-- HDF5 files are inside a timestamped folder!
│   │   │   │   ├── backside_external_sensor_signals.h5
│   │   │   │   ├── backside_internal_machine_signals.h5
The script is directly looking in:
cylinder_bottom/cnc_milling_machine/process_data/
# 🔹 Purpose: Efficiently inspect HDF5 files without full loading
# 🔹 Features:
#    - Lists available datasets inside an HDF5 file
#    - Retrieves metadata (shape, dtype) for each dataset
#    - Loads a small sample of data to preview content
"""
# ============================================================
# 1️⃣ Define HDF5 File Paths
# ============================================================
DATA_ROOT = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\raw"
hdf5_files = [
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "backside_external_sensor_signals.h5"),
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "backside_internal_machine_signals.h5"),
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "frontside_external_sensor_signals.h5"),
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "frontside_internal_machine_signals.h5"),
]


# ============================================================


"""🚀 Step 1: Modify the File Scanner
1️⃣ HDF5 files (.h5) ✅ Already handled
2️⃣ Metadata files (meta_data.json)
3️⃣ Quality control data (quality_data.csv)
4️⃣ Timestamp mappings (*_timestamp_process_pairs.csv """

# ============================================================

# import os

# # 📌 Define root directory
# DATA_ROOT = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\raw"

# 📌 Function to find all relevant files (HDF5, JSON, CSV)
def find_all_files(root_dir):
    """
    📌 Scans the directory recursively to find:
    - HDF5 process data files (*.h5)
    - Metadata files (meta_data.json)
    - Quality control data (quality_data.csv)
    - Timestamp mappings (*_timestamp_process_pairs.csv)
    - Production logs (*.xlsx)
    
    🔹 Returns:
        - hdf5_files (list): Paths of all HDF5 files
        - metadata_files (list): Paths of metadata JSON files
        - quality_files (list): Paths of quality CSV files
        - timestamp_files (list): Paths of timestamp mapping CSV files
        - production_log_files (list): Paths of production log Excel files
    """
    hdf5_files = []
    metadata_files = []
    quality_files = []
    timestamp_files = []
    production_log_files = []  # New list to store production log files

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".h5"):  
                hdf5_files.append(file_path)  # Store HDF5 process data files
            elif file == "meta_data.json":  
                metadata_files.append(file_path)  # Store metadata files
            elif file == "quality_data.csv":  
                quality_files.append(file_path)  # Store quality control files
            elif "timestamp_process_pairs.csv" in file:  
                timestamp_files.append(file_path)  # Store timestamp mappings
            elif file.endswith(".xlsx") and "production_log" in file:  
                production_log_files.append(file_path)  # Store production log files

    return hdf5_files, metadata_files, quality_files, timestamp_files, production_log_files

# 🔹 Scan the dataset
hdf5_files, metadata_files, quality_files, timestamp_files, production_log_files = find_all_files(DATA_ROOT)

# ✅ Print summary of findings
print("\n📊 **File Scan Summary**")
print(f"✅ Found {len(hdf5_files):,} HDF5 Process Data Files")
print(f"✅ Found {len(metadata_files):,} Metadata Files")
print(f"✅ Found {len(quality_files):,} Quality Control CSVs")
print(f"✅ Found {len(timestamp_files):,} Timestamp Mapping CSVs")
print(f"✅ Found {len(production_log_files):,} Production Log Excel Files")


#-------------------------------------------------------------

#-------------------------------------------------------------

"""🚀 Step 2: Verify the Metadata (meta_data.json)
Now that we've found the metadata files, let's inspect them to ensure they contain the expected 
component and process information."""

#-------------------------------------------------------------

import json

# 📌 Function to inspect metadata files correctly
def inspect_metadata(file_path):
    """Reads and displays key contents of a metadata JSON file, handling list-based structures."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)  # Load JSON file

        # ✅ Ensure it's a list and iterate through the items
        if isinstance(data, list):
            print(f"\n📂 Inspecting Metadata File: {file_path}")
            print(f"🔹 Total Entries: {len(data)}")

            # Process the first entry for preview (avoid overwhelming output)
            first_entry = data[0] if data else {}

            print(f"🔹 Sample Entry Keys: {list(first_entry.keys())}")
            print(f"🔹 Part Type: {first_entry.get('part_type', 'N/A')}")
            print(f"🔹 Part ID: {first_entry.get('part_id', 'N/A')}")
            print(f"🔹 Number of Process Data Files: {len(first_entry.get('process_data', []))}")
            print(f"🔹 Number of Quality Data Entries: {len(first_entry.get('quality_data', []))}\n")

        else:
            print(f"\n⚠️ Unexpected JSON structure in {file_path}. Expected a list but found {type(data)}.")

    except Exception as e:
        print(f"❌ Error reading metadata file {file_path}: {e}")

# 🔹 Run the inspection again
for meta_file in metadata_files:
    inspect_metadata(meta_file)

#-------------------------------------------------------------

"""🚀 Step 3: Verify Quality Data (quality_data.csv)
Let's check the quality control data, which contains measurement results for manufactured parts."""

#-------------------------------------------------------------


# 📌 Function to inspect quality control CSVs
def inspect_quality_data(file_path):
    """Loads and previews the first few rows of a quality control CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"\n📂 Inspecting Quality Data File: {file_path}")
        print(df.head())  # Display first few rows

    except Exception as e:
        print(f"❌ Error reading quality data file {file_path}: {e}")

# 🔹 Inspect all quality data files

print("\n📊 **Quality Data Inspection**")
for quality_file in quality_files:
    inspect_quality_data(quality_file)
    
    
#-------------------------------------------------------------
"""🚀 Step 4: Verify Timestamp Mappings (*_timestamp_process_pairs.csv)
These files map timestamps to specific subprocesses, so we need to ensure they are structured correctly."""

#-------------------------------------------------------------
# 📌 Function to inspect timestamp mapping files
def inspect_timestamp_mappings(file_path):
    """Loads and previews the first few rows of a timestamp mapping CSV file."""
    try:
        df = pd.read_csv(file_path, names=["timestamp", "process"])
        print(f"\n📂 Inspecting Timestamp Mapping File: {file_path}")
        print(df.head())  # Display first few rows

    except Exception as e:
        print(f"❌ Error reading timestamp mapping file {file_path}: {e}")

# 🔹 Inspect all timestamp mapping files
print("\n📊 **Timestamp Mapping Inspection**"   )
for timestamp_file in timestamp_files:
    inspect_timestamp_mappings(timestamp_file)

#-------------------------------------------------------------
# ============================================================
# 🚀 HDF5 Data Extraction and Conversion Utility
#Extract Specific Data from HDF5  and Convert HDF5 Data into Pandas
# ============================================================
# 🔹 Purpose: Extract specific datasets from HDF5 files efficiently.
# 🔹 Features:
#    - Lists all datasets dynamically
#    - Loads only a specific dataset (avoiding memory overload)
#    - Converts extracted data into Pandas DataFrame
# ============================================================

# 1️⃣ Define HDF5 File Paths

DATA_ROOT = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\raw"

hdf5_files = [
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "backside_external_sensor_signals.h5"),
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "backside_internal_machine_signals.h5"),
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "frontside_external_sensor_signals.h5"),
    os.path.join(DATA_ROOT, "cylinder_bottom", "cnc_milling_machine", "process_data", "100101_11_29_2022_12_30_29", "frontside_internal_machine_signals.h5"),
]

# 2️⃣ Function to Extract Data from HDF5

def extract_hdf5_data(file_path, dataset_name="data", sample_size=5):
    """
    📌 Extracts a specific dataset from an HDF5 file and converts it to a Pandas DataFrame.

    🔹 Parameters:
        - file_path (str): Path to the HDF5 file
        - dataset_name (str): Name of the dataset to extract (default: "data")
        - sample_size (int): Number of rows to load (default: 5)

    🔹 Steps:
        1. Opens the HDF5 file in read mode
        2. Checks if the dataset exists
        3. Loads a small sample (e.g., first 5 rows)
        4. Converts extracted data into a Pandas DataFrame

    🔹 Why?
        - Avoids full memory loading
        - Enables easy Pandas analysis
    """

    # ✅ Step 1: Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ HDF5 File NOT Found: {file_path}")
        return None
    
    try:
        # ✅ Step 2: Open HDF5 file in read mode
        with h5py.File(file_path, "r") as h5f:
            # ✅ Step 3: List available datasets dynamically
            datasets = list(h5f.keys())
            print(f"\n🔍 Available Datasets in {file_path}: {datasets}")

            if dataset_name not in datasets:
                print(f"⚠️ Dataset '{dataset_name}' not found in {file_path}. Skipping...")
                return None

            # ✅ Step 4: Extract a small sample of data
            dataset = h5f[dataset_name]
            data_sample = dataset[:sample_size]  # Load first 'sample_size' rows

            # ✅ Step 5: Convert to Pandas DataFrame
            df = pd.DataFrame(data_sample)

            print(f"✅ Extracted '{dataset_name}' from {file_path} - Shape: {df.shape}")
            return df

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

# 3️⃣ Extract and Convert HDF5 Data to Pandas

print("\n🚀 Extracting Data from HDF5 Files...\n")

# Loop through each file and extract data
for hdf5_path in hdf5_files:
    extracted_df = extract_hdf5_data(hdf5_path, dataset_name="data", sample_size=10)  # Load first 10 rows
    if extracted_df is not None:
        print(extracted_df.head())  # Display first few rows

print("\n🎉 HDF5 Data Extraction Completed Successfully!")

extracted_df.info()
extracted_df.describe()
extracted_df.columns
extracted_df.shape
extracted_df.raw

#-------------------------------------------------------------
"""📌 Code: Extract First 10 Rows from Each Dataset in HDF5 Files
To extract the first 10 rows from each dataset inside the HDF5 files across all process data directories, we need to:

1️⃣ Recursively search for .h5 files in all process data folders (cylinder_bottom, saw, piston_rod, etc.).
2️⃣ Inspect available datasets inside each .h5 file.
3️⃣ Extract the first 10 rows from each dataset inside the HDF5 files.
4️⃣ Store and display the extracted data to understand the structure and content.

# 📂 Define the root directory where the raw data is stored
DATA_ROOT = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\raw" """

process_data_root = r"C:\DT_Projects\Unified-AI-Pipeline-for-Predictive-Maintenance-and-Optimization\data\raw"

# Function to find all HDF5 files in subdirectories
def find_hdf5_files(root_dir):
    hdf5_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".h5"):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

# Find all HDF5 files in the process_data directory
hdf5_files = find_hdf5_files(process_data_root)

# Print found files
if hdf5_files:
    print("\n✅ Found HDF5 Files:")
    for file in hdf5_files:
        print(file)
else:
    print("\n❌ No HDF5 files found in the specified directory!")

# 📌 Function to extract the first 10 rows from each dataset inside an HDF5 file
def extract_hdf5_samples(file_path, num_rows=10):
    """Extracts the first `num_rows` rows from each dataset in the given HDF5 file."""
    try:
        with h5py.File(file_path, "r") as h5f:
            print(f"\n============================================================")
            print(f"📂 Inspecting HDF5 File: {file_path}")
            print(f"============================================================")

            # 🔍 List all available datasets in the file
            dataset_names = list(h5f.keys())
            print(f"🔍 Available Datasets: {dataset_names}")

            # 📤 Extract the first `num_rows` rows from each dataset
            for dataset_name in dataset_names:
                data = h5f[dataset_name][:num_rows]  # Extract first `num_rows` rows
                df_sample = pd.DataFrame(data)  # Convert to Pandas DataFrame

                print(f"\n🔹 Dataset: {dataset_name}")
                print(f"   ├─ Shape: {h5f[dataset_name].shape}")
                print(f"   ├─ Data Type: {h5f[dataset_name].dtype}")
                print(f"   ├─ Sample Data (First {num_rows} Rows):")
                print(df_sample)  # Print extracted sample data

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

# 🔍 Find all HDF5 files in process_data directories
hdf5_files = find_hdf5_files(DATA_ROOT)

# 🔽 Extract and display first 10 rows from each dataset inside each HDF5 file
for h5_file in hdf5_files:
    extract_hdf5_samples(h5_file, num_rows=10)
    
#-------------------------------------------------------------

#statistics value of hdfs files
# 📌 Function to extract basic statistics from HDF5 files

def extract_hdf5_stats(file_path):
    """Extracts basic statistics (shape, data type) for each dataset in the given HDF5 file."""
    try:
        with h5py.File(file_path, "r") as h5f:
            print(f"\n============================================================")
            print(f"📂 Inspecting HDF5 File: {file_path}")
            print(f"============================================================")

            # 🔍 List all available datasets in the file
            dataset_names = list(h5f.keys())
            print(f"🔍 Available Datasets: {dataset_names}")

            # 📊 Extract basic statistics for each dataset
            for dataset_name in dataset_names:
                dataset = h5f[dataset_name]
                print(f"\n🔹 Dataset: {dataset_name}")
                print(f"   ├─ Shape: {dataset.shape}")
                print(f"   ├─ Data Type: {dataset.dtype}")

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        
print("\n📊 **HDF5 Statistics Extraction**"
      # Extract and display basic statistics for each dataset in all HDF5 files)
for h5_file in hdf5_files:
    extract_hdf5_stats(h5_file)
    
    
#-------------------------------------------------------------


#================================================================
