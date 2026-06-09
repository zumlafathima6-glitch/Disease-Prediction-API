from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")
breast_model = joblib.load("breast_model.pkl")
parkinsons_model = joblib.load("parkinsons_model.pkl")


class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int


class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class BreastInput(BaseModel):
    features: list[float]  # 30 features from WDBC dataset


class ParkinsonsInput(BaseModel):
    fo: float             # Average vocal fundamental frequency
    fhi: float            # Maximum vocal fundamental frequency
    flo: float            # Minimum vocal fundamental frequency
    jitter_percent: float # Jitter (%)
    jitter_abs: float     # Jitter (Abs)
    rap: float            # RAP
    ppq: float            # PPQ
    ddp: float            # DDP
    shimmer: float        # Shimmer
    shimmer_db: float     # Shimmer (dB)
    apq3: float           # APQ3
    apq5: float           # APQ5
    apq: float            # APQ
    dda: float            # DDA
    nhr: float            # NHR
    hnr: float            # HNR
    rpde: float           # RPDE
    dfa: float            # DFA
    spread1: float        # Spread1
    spread2: float        # Spread2
    d2: float             # D2
    ppe: float            # PPE


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        features = np.array([[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree_function,
            data.age
        ]])
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        features = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]])
        prediction = heart_model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/breast")
def predict_breast(data: BreastInput):
    try:
        input_data = np.array(data.features).reshape(1, -1)

        if input_data.shape[1] != 30:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 30 features, got {input_data.shape[1]}"
            )

        prediction = breast_model.predict(input_data)[0]

        confidence = None
        if hasattr(breast_model, "predict_proba"):
            confidence = float(max(breast_model.predict_proba(input_data)[0]))

        return {
            "prediction": int(prediction),
            "result": "Malignant" if prediction == 1 else "Benign",
            "confidence": confidence
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/parkinsons")
def predict_parkinsons(data: ParkinsonsInput):
    try:
        features = np.array([[
            data.fo, data.fhi, data.flo,
            data.jitter_percent, data.jitter_abs,
            data.rap, data.ppq, data.ddp,
            data.shimmer, data.shimmer_db,
            data.apq3, data.apq5, data.apq, data.dda,
            data.nhr, data.hnr,
            data.rpde, data.dfa,
            data.spread1, data.spread2,
            data.d2, data.ppe
        ]])

        prediction = parkinsons_model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "result": "Parkinson's Detected" if prediction == 1 else "Healthy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))