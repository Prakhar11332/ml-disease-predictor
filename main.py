from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Predictor API")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + scaler at startup
model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# This defines the shape of the input JSON
class PatientData(BaseModel):
    age: float
    sex: float         # 0=female, 1=male
    cp: float          # chest pain type 0-3
    trestbps: float    # resting blood pressure
    chol: float        # cholesterol
    fbs: float         # fasting blood sugar > 120mg (0/1)
    restecg: float     # resting ECG 0-2
    thalach: float     # max heart rate achieved
    exang: float       # exercise-induced angina (0/1)
    oldpeak: float     # ST depression
    slope: float       # slope of peak exercise ST
    ca: float          # number of major vessels 0-3
    thal: float        # 3=normal, 6=fixed defect, 7=reversible

@app.get("/")
def root():
    return {"message": "Heart Disease Predictor API is running"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to numpy array in correct feature order
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]])
    
    # Scale using same scaler as training
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        "prediction": int(prediction),
        "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK",
        "probability": round(float(probability), 4),
        "confidence": f"{probability*100:.1f}%"
    }