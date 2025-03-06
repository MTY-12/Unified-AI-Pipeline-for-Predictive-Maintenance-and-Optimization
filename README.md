# Unified AI Pipeline for Predictive Maintenance and Optimization 🚀

## Overview
This project develops a unified AI pipeline to address key challenges in discrete manufacturing, including:
- Preprocessing
- Condition Monitoring
- Tool Wear Prediction
- Remaining Useful Life (RUL) Estimation
- Predictive Maintenance
- Process Optimization
- Bottleneck Detection

By leveraging sensor data and advanced AI models, this project aims to optimize manufacturing processes, reduce downtime, and enhance efficiency.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Key Techniques](#key-techniques)
4. [Dataset](#dataset)
5. [Workflow](#workflow)
6. [Setup and Usage](#setup-and-usage)
7. [Conda Environment Setup](#conda-environment-setup)
8. [Results](#results)
9. [Contributors](#contributors)

---

## Features ✨
- **Preprocessing**: Standardize and clean sensor data for analysis.
- **Condition Monitoring**: Monitor real-time equipment health and detect anomalies.
- **Tool Wear Prediction**: Predict tool wear to optimize replacement schedules.
- **RUL Estimation**: Estimate the remaining useful life of components.
- **Predictive Maintenance**: Predict and mitigate failures to reduce downtime.
- **Process Optimization**: Streamline workflows to maximize efficiency.
- **Bottleneck Detection**: Identify and resolve process bottlenecks.

---

## Key Techniques 🔍
- **Data Preprocessing**: Handle noise, outliers, and feature engineering.
- **Neural Networks**: Build models for prediction tasks.
- **Clustering Techniques**: Analyze trends and group data.
- **Regression Models**: Predict tool wear and remaining useful life.
- **Classification Models**: Categorize equipment conditions (e.g., normal, urgent).
- **Evaluation Metrics**: Use RMSE, Precision, Recall, and F1-score to measure model performance.

---

## Dataset 📂
This project uses sensor data collected from manufacturing equipment, which includes:
- **Raw Data**: Temperature, pressure, vibration, speed, and metadata.
- **Processed Data**: Cleaned and normalized for machine learning input.

---

## Workflow 🛠️
### 1. Preprocessing
- Standardize and clean raw sensor data.
- Handle inconsistencies and apply feature engineering.

### 2. Condition Monitoring
- Build dashboards for real-time and historical equipment health visualization.
- Use unsupervised learning techniques to detect anomalies.

### 3. Tool Wear Prediction
- Develop regression models to predict tool wear.
- Use operational data like time, force, and temperature.

### 4. RUL Estimation
- Implement supervised learning models (LSTMs, Transformers).
- Predict remaining useful life using historical sensor data.

### 5. Predictive Maintenance
- Train models to forecast potential failures.
- Use cost-sensitive training to minimize critical mistakes.

### 6. Process Optimization
- Use predictive insights to streamline workflows.
- Perform what-if simulations to evaluate changes.

### 7. Bottleneck Detection
- Analyze production line data to detect delays.
- Apply clustering to identify and resolve constraints.

---

## Setup and Usage ⚙️
### Prerequisites
- Python 3.9+
- Required Libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `torch`, `matplotlib`, `seaborn`

### Installation
```bash
git clone https://github.com/<MTY-12 >/unified-ai-pipeline.git
cd unified-ai-pipeline
pip install -r requirements.txt
```

### Conda Environment Setup
1. Create environment:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate environment:
    ```bash
    conda activate unified-ai-pipeline
    ```
3. Update environment:
    ```bash
    conda env update -f environment.yml --prune
    ```

---

## Results 📊
The pipeline delivers:
- **Preprocessed Dataset**: Ready for modeling.
- **Condition Monitoring Dashboard**: Real-time and historical visualizations.
- **Tool Wear Prediction Model**: Optimized tool replacement schedules.
- **RUL Estimation Framework**: Accurate component lifespan predictions.
- **Predictive Maintenance System**: Reduced downtime and optimized scheduling.
- **Process Optimization Insights**: Workflow and efficiency improvements.
- **Bottleneck Analysis Report**: Identified constraints and recommended actions.

---

## Contributors 👥
- **[MTY]**: DTE
- Open for Collaboration! Feel free to contribute to this repository.

---

## License 📜
This project is licensed under the MTY License. See the `LICENSE` file for details.

---

## Contact 📬
- Email: mty_12@outlook.com
- LinkedIn: 
