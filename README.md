# 🫀 Heart Disease Risk Predictor

A machine learning web application that predicts heart disease risk from clinical patient data — returning a risk label, probability score, and SHAP-based explanation of which features drove the prediction.

**Live Demo → [https://ml-disease-predictor-lrre.onrender.com](https://ml-disease-predictor-lrre.onrender.com)**

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death globally. Early detection using clinical data can significantly improve patient outcomes. This project builds an end-to-end ML pipeline that takes 13 clinical features from a patient and predicts whether they are at risk of heart disease — along with a probability score and a feature-level explanation of why.

---

## 📊 Dataset

**Heart Disease UCI — Cleveland Dataset**

| Property | Details |
|---|---|
| Source | UCI Machine Learning Repository |
| Patients | 303 |
| Features | 13 clinical features |
| Target | Binary — 0 (No Disease), 1 (Disease) |
| Class balance | ~54% disease, ~46% no disease |

### Features used

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels coloured by fluoroscopy (0–3) |
| thal | Thalassemia type (3 = normal, 6 = fixed defect, 7 = reversible) |

---

## 🔍 EDA Findings

- **thalach** (max heart rate) showed the strongest negative correlation with heart disease — patients with disease had significantly lower max heart rates
- **cp** (chest pain type) was highly predictive — type 0 (asymptomatic) was most associated with disease presence
- **ca** (number of vessels) and **oldpeak** (ST depression) both showed clear separation between disease and non-disease groups in box plots
- No missing values in the dataset — no imputation required
- Class distribution was balanced (~54/46) — no SMOTE or resampling needed

---

## 🤖 Model Comparison

Four models were trained and evaluated on the same 80/20 stratified train-test split with StandardScaler preprocessing:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 80% | 87 |
| **Random Forest** ✅ | **~83%** | **91** |
| SVM | 82% | 88 |


**Random Forest** was selected as the final model based on highest accuracy and strong ROC-AUC. 5-fold cross-validation confirmed stability.

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to explain model predictions at both the global and individual level.

- **Summary plot** — shows which features have the most impact across all test patients. `sex`, `age` consistently ranked as the top 2 most impactful features.
- **Waterfall plot** — shows for a single patient exactly which features pushed the prediction towards or away from disease, and by how much.

> SHAP plots are saved in the repo under `outputs/`.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| ML models | scikit-learn, XGBoost |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Model persistence | joblib |
| Frontend | HTML, CSS, JavaScript (Fetch API) |
| Deployment | Render (backend), GitHub Pages (frontend) |

---

## 🗂️ Project Structure

```
ml-disease-predictor/
├── main.py              ← FastAPI app with /predict endpoint
├── model.joblib         ← Trained Random Forest model
├── scaler.joblib        ← Fitted StandardScaler
├── requirements.txt     ← Python dependencies
├── index.html           ← Frontend UI
├── .gitignore
└── outputs/
    ├── shap_summary.png
    └── shap_waterfall.png
```

---

## 🚀 API Reference

**Base URL:** `https://ml-disease-predictor-lrre.onrender.com`

### `GET /`
Health check — returns API status.

### `POST /predict`
Accepts patient data and returns prediction.

**Request body:**
```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

**Response:**
```json
{
  "prediction": 1,
  "risk_label": "HIGH RISK",
  "probability": 0.7734,
  "confidence": "77.3%"
}
```

---

## 🖥️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Prakhar11332/ml-disease-predictor.git
cd ml-disease-predictor

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API
uvicorn main:app --reload

# 5. Open in browser
# API docs → http://localhost:8000/docs
# Frontend → open index.html in browser
```

---

## 👤 Author

**Prakhar**
B.Tech Computer Science (AI Specialization) — Amrita Vishwa Vidyapeetham

[GitHub](https://github.com/Prakhar11332) · [Live Demo](https://ml-disease-predictor-lrre.onrender.com)
