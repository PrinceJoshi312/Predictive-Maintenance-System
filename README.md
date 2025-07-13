# 🛠️ Predictive Maintenance System for Centrifugal Pumps

This project implements a **predictive maintenance system** using **sensor data from centrifugal pumps**, combining traditional **machine learning** and **deep learning** models to classify pump health and prevent failures in an industrial setting.

---

## 📊 Dataset

* **Source**: Kaggle
* **File**: `Centrifugal_pumps_measurements.xlsx`
* **Description**: Contains real-time sensor data from two centrifugal pumps labeled by machine ID (1: Healthy, 2: Maintenance-prone).

---

## 🧪 Workflow Summary

### 🔹 Data Cleaning & Preprocessing

* Removed missing values and outliers (via IQR method)
* Verified no duplicates
* Categorical date-time features converted to cyclical representations (sin/cos)
* Created lagged and interaction features
* Final features: 16 engineered inputs for prediction

### 🔹 Feature Engineering

* Cyclical features: `minute_sin`, `hour_cos`, etc.
* Rolling stats: `value_ISO_rolling_mean`, `value_ISO_rolling_std`
* Lags: `value_ISO_lag1`, `value_ISO_lag2`
* Composite features: `value_ISO_TEMP = value_ISO * valueTEMP`

### 🔹 Label Creation

* Target: `health_status` (0 = Healthy, 1 = Maintenance-Prone)

---

## 🧠 Models Used

### ✅ Random Forest Classifier

* Accuracy: **100%**
* Served as a baseline model

### ✅ Support Vector Machine (SVM)

* Accuracy: **95.8%**
* Implemented via a `Pipeline` with `StandardScaler`

### ✅ Deep Neural Network

* Sequential model with 3 hidden layers
* Accuracy: **94.5%**
* Architecture:

  * Input → Dense(64) → Dense(32) → Dense(16) → Output(sigmoid)

### ✅ Ensemble (SVM + DNN)

* Final prediction = average of SVM and DNN probabilities
* Accuracy: **95.8%**
* Balanced precision and recall for both classes

---

## 🧮 Sample Prediction

```python
X_new = [[0.5, 0.8, 75, 0.6, 0.9, 0.2, 0.1, 0.7, 0.5, 0.8, 1, 0, 0.5, 0.3, 0.4, 0.6]]
```

### Output:

```
SVM Probability:         0.0779
Neural Net Probability:  0.0000
Combined:                0.0389 → Prediction: Healthy (0)
```

---

## 🧾 Evaluation Metrics

* **Accuracy**, **Precision**, **Recall**, **F1-score**
* **Confusion Matrix** used to verify true positives/negatives

---

## 🧳 Model Export

* SVM model: `svm_pipeline.pkl` (via `joblib`)
* Neural network: `dl_model.h5` (via `TensorFlow`)

You can load and use them for future predictions using:

```python
svm_pipeline = joblib.load('svm_pipeline.pkl')
dl_model = load_model('dl_model.h5')
```

---

## 📦 Dependencies

Install required libraries with:

```bash
pip install -r requirements.txt
```

Main libraries:

* `pandas`, `numpy`, `seaborn`, `matplotlib`
* `scikit-learn`, `tensorflow`, `joblib`

---

## 📌 Highlights

* Industrial-grade feature engineering
* Ensemble strategy to increase robustness
* Can be integrated into real-time monitoring systems

---

## 👨‍💻 Author

**Prince Joshi**
GitHub: [@PrinceJoshi312](https://github.com/PrinceJoshi312)
