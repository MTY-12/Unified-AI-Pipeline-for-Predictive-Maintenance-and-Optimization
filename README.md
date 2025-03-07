
# Unified AI Pipeline for Anomaly Detection and Predictive Maintenance 🚀

## Overview
Anomaly detection plays a crucial role in predictive maintenance by identifying early signs of equipment failure, enabling proactive interventions to reduce downtime and costs. This project develops a **Unified AI Pipeline** that integrates **deep learning** (Autoencoders), **classical machine learning models** (Random Forest, One-Class SVM, K-Nearest Neighbors), and **ensemble learning** for **robust anomaly detection** in manufacturing processes.

The pipeline utilizes the **CiP-DMD dataset**, which contains sensor data from a real-world **multi-step machining process**. By leveraging advanced **dimensionality reduction (PCA)** and **ensemble learning**, the project aims to **improve detection accuracy and optimize predictive maintenance strategies**.

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
- **AI-Powered Anomaly Detection**: Combination of **Autoencoders, Random Forest, One-Class SVM, and KNN**.
- **Ensemble Learning**: Improves accuracy by combining predictions from multiple models.
- **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining **95% of data variance**.
- **Custom Feature Engineering**: Extracts condition indicators from **sensor data**.
- **Optimized Anomaly Thresholding**: Balances **Precision, Recall, and F1-score** for optimal classification.
- **Scalability**: Designed for real-time industrial deployment with **SCADA integration** (future work).

---

## Key Techniques 🔍
1. **Principal Component Analysis (PCA)** for **feature extraction** and dimensionality reduction.
2. **Autoencoder-based anomaly detection** to reconstruct normal operational patterns.
3. **Random Forest Classifier** for robust pattern recognition in sensor data.
4. **One-Class SVM** to detect outliers in unsupervised settings.
5. **K-Nearest Neighbors (KNN)** using **Manhattan, Minkowski, and Cosine distance metrics** for adaptive anomaly detection.
6. **Ensemble Learning Approach** to integrate multiple models using **majority voting**.

---

## Dataset 📂
This project uses the **CiP-DMD dataset**, an **open-source industrial dataset** that contains sensor readings from **847 pneumatic cylinders**.  
[**Dataset Link**](https://cloud.ptw-darmstadt.de/index.php/s/5Wv34VRZEXBLsZK?path=%2F)

### Data Components:
- **Raw Data**: Includes temperature, pressure, vibration, speed, and metadata.
- **Processed Data**: Cleaned and normalized for AI model training.
- **Annotations**: Metadata includes **quality control labels and machine logs**.

![Dataset](https://github.com/user-attachments/assets/6b2c9f18-fe7a-442b-9b89-16362339b35d)

---

## Workflow 🛠️
The pipeline follows a **structured AI-driven approach**:

### 1️⃣ Data Acquisition
- **Extract data** from **HDF5 files** containing over **85 million sensor readings**.
- **Handle missing values, outliers, and format inconsistencies**.

### 2️⃣ Feature Engineering & Transformation
- **Principal Component Analysis (PCA)** to optimize **feature representation**.
- **Normalization and scaling** for improved model performance.

### 3️⃣ AI Model Development
- **Phase 1: Deep Learning-based Anomaly Detection**  
  - **Autoencoder** trained exclusively on normal data.
  - **Reconstruction error thresholding** for anomaly classification.
  ![image](https://github.com/user-attachments/assets/d992c3cf-1c33-4e2a-afd7-c94fe5fe6656)

- **Phase 2: Classical ML Anomaly Classification**
  - **Random Forest**, **One-Class SVM**, and **KNN** (various distance metrics).
  - Recursive Feature Elimination (RFE) for selecting the **most important features**.

- **Phase 3: Ensemble Learning**
  - **Majority voting approach** to combine model predictions.
  - Improves **recall and robustness** against false positives.

---

## Setup and Usage ⚙️
### Prerequisites
- Python **3.9+**
- Required Libraries:  
  ```bash
  pip install pandas numpy scikit-learn tensorflow torch matplotlib seaborn
  ```

### Installation
```bash
git clone https://github.com/MTY-12/unified-ai-pipeline.git
cd unified-ai-pipeline
pip install -r requirements.txt
```

---

## Conda Environment Setup
1. **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
2. **Activate the environment:**
    ```bash
    conda activate unified-ai-pipeline
    ```
3. **Update the environment:**
    ```bash
    conda env update -f environment.yml --prune
    ```

---

## Results 📊
### 3D Visualization of Ensemble Model Predictions
The **ensemble learning approach** successfully classified anomalies in the dataset, as shown in the 3D PCA representation:

![3D Scatter Plot](https://github.com/user-attachments/assets/73e7166d-728b-454b-af58-ce79d692e4d3)

- **Red points** indicate detected anomalies.
- **Blue points** indicate normal operations.
- **The model effectively separates anomalous behavior** but requires further refinement for edge cases.

### Feature Importance Ranking  
![image](https://github.com/user-attachments/assets/55d3c3fc-74f9-431e-984a-507f4850371f)

The **most significant features** contributing to anomaly detection include:
- **KNN predictions (Minkowski, Euclidean, Manhattan)**
- **PCA components (PC1, PC2, PC18)**
- **Autoencoder-based anomaly scores**
- **Optimized ensemble predictions**

![Precision, Recall, and F1-Score change as we lower the anomaly detection threshold from 95% to 60%.]
![image](https://github.com/user-attachments/assets/444b154d-8f76-432e-85a2-e9b4f95af6e3)

---

## Contributors 👥
- **[Michael Yerdaw]**: Digital Engineering Student, AI & Automation Engineer  
  - 🔬 Researching **AI-powered predictive maintenance**  
  - 🔧 Building **SCADA-integrated AI solutions**  
  - 📢 Open for collaboration!  

---

## License 📜
This project is **licensed under the MTY License**. See the `LICENSE` file for details.

---

## Contact 📬
- 📧 Email: **mty_12@outlook.com**
- 🔗 LinkedIn: 
- 🌍 GitHub: **[https://github.com/MTY-12]**

---

### **AI Usage Transparency**
This project **incorporates AI-assisted development**, and to reflect that, I am marking this assignment under the **"CYBORG"** category from the **Me & My Machine framework**.

#### **AI Contribution Includes**:
- **Refining technical explanations** via AI models.
- **Extracting structured content** from research papers.
- **Optimizing ML pipelines** with auto-generated code suggestions.
- **Iterative AI-driven debugging and performance tuning**.


---

## References 📚
1. **Breunig, M. M., et al.** (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD*.  
   - [🔗 Paper](https://dl.acm.org/doi/10.1145/335191.335388)
   
2. **Chandola, V., et al.** (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*.  
   - [🔗 Paper](https://dl.acm.org/doi/10.1145/1541880.1541882)
   
3. **Aggarwal, C. C.** (2013). Outlier Analysis. *Springer*.  
   - [🔗 Book](https://link.springer.com/book/10.1007/978-1-4614-6396-2)
   
4. **Baglee, D., & Knowles, M.** (2010). Maintenance strategy development within SME manufacturing organizations. *Journal of Quality in Maintenance Engineering*.  
   - [🔗 Paper](https://doi.org/10.1115/DETC2007-35920)

---

### 🔥 *"AI is not just a tool; it’s a collaborative force in engineering innovation!"* 🚀

---
