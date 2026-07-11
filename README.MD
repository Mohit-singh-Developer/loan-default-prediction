# 🏦 Loan Default Prediction using Machine Learning

An end-to-end Machine Learning project that predicts the probability of loan default using customer demographic and financial information. The project includes data preprocessing, feature engineering, model training, explainability using SHAP, and deployment through a Streamlit web application.

---

## 📌 Project Overview

Financial institutions face significant losses due to loan defaults. This project aims to identify high-risk applicants before loan approval by leveraging machine learning techniques on the Home Credit Default Risk dataset.

The application allows users to enter customer details and predicts the probability of loan default in real time.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Comparison
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter Tuning
- Model Explainability using SHAP
- Streamlit Web Application
- Real-time Loan Default Prediction

---

## 📊 Dataset

- **Dataset:** Home Credit Default Risk
- **Source:** Kaggle
- **Records:** 307,511 Loan Applications
- **Features:** 121 Original Features

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib
- Matplotlib

---

## ⚙️ Feature Engineering

The following business-oriented features were created:

- Credit-to-Income Ratio
- Credit-to-Goods Ratio
- Employment-to-Age Ratio

These engineered features improved the model's ability to distinguish between low-risk and high-risk applicants.

---

## 🤖 Models Evaluated

| Model | Evaluation Metric |
|--------|-------------------|
| Logistic Regression | ROC-AUC |
| Random Forest | ROC-AUC |
| XGBoost | ROC-AUC |

The final deployed model uses **XGBoost** due to its superior predictive performance.

---

## 🖥️ Streamlit Application

The application allows users to:

- Enter customer financial information
- Predict default probability
- View risk level
- Receive loan recommendation
- Review financial ratios

---

## 📂 Project Structure

```text
loan-default-prediction/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│   └── deployment_model.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Installation

Clone the repository

```bash
git clone <your-repository-link>
```

Move into the project directory

```bash
cd loan-default-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## 🔮 Future Improvements

- Deploy the application on Streamlit Community Cloud
- Improve probability calibration
- Incorporate additional financial and credit history features
- Enhance the user interface and reporting dashboard

---

## 👨‍💻 Author

**Mohit Singh**

B.Tech – Electronics and Communication Engineering

IIITDM Jabalpur

GitHub: https://github.com/<your-username>

LinkedIn: https://linkedin.com/in/<your-profile>