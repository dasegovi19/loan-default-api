# app.py

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# 1. Load your saved model
#model = joblib.load("models/best_model.joblib")

import os
import joblib
import gdown

# 1. Where to store the downloaded model
MODEL_PATH = "models/best_model.joblib"
# 2. File ID from your Drive share link
DRIVE_ID = "1_vdAJ08J12XK7eVzro-tVnfrI6EJqh1I"

# 3. Download the model if it's not already on disk
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# 4. Load the model into memory
model = joblib.load(MODEL_PATH)



# 2. Declare the FastAPI “app” object
app = FastAPI()

# 3. Define request schema
class LoanFeatures(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    DTI: float

    Education_High_School: int = Field(0, alias='Education_High School')
    Education_Masters: int     = Field(0, alias="Education_Master's")
    Education_PhD: int         = Field(0, alias='Education_PhD')

    EmploymentType_Part_time: int     = Field(0, alias='EmploymentType_Part-time')
    EmploymentType_Self_employed: int = Field(0, alias='EmploymentType_Self-employed')
    EmploymentType_Unemployed: int    = Field(0, alias='EmploymentType_Unemployed')

    MaritalStatus_Married: int = Field(0, alias='MaritalStatus_Married')
    MaritalStatus_Single: int  = Field(0, alias='MaritalStatus_Single')

    HasMortgage_Yes: int       = Field(0, alias='HasMortgage_Yes')
    HasDependents_Yes: int     = Field(0, alias='HasDependents_Yes')

    LoanPurpose_Business: int  = Field(0, alias='LoanPurpose_Business')
    LoanPurpose_Education: int = Field(0, alias='LoanPurpose_Education')
    LoanPurpose_Home: int      = Field(0, alias='LoanPurpose_Home')
    LoanPurpose_Other: int     = Field(0, alias='LoanPurpose_Other')

    HasCoSigner_Yes: int       = Field(0, alias='HasCoSigner_Yes')

    FICO_Band_Low: int         = Field(0, alias='FICO_Band_Low')
    FICO_Band_Medium: int      = Field(0, alias='FICO_Band_Medium')
    FICO_Band_High: int        = Field(0, alias='FICO_Band_High')
    FICO_Band_VeryHigh: int    = Field(0, alias='FICO_Band_VeryHigh')

    class Config:
        validate_by_name = True  # Pydantic v2 syntax

# 4. Register a POST endpoint
@app.post("/predict")
def predict_default(features: LoanFeatures):
    df = pd.DataFrame([features.dict(by_alias=True)])
    df = df[model.feature_names_in_]  # Ensure feature alignment
    prob = model.predict_proba(df)[0, 1]
    return {"default_probability": prob}







