Loan Default Prediction API

 

📑 Table of Contents

Overview

Demo

Repository Structure

Setup

Data Preparation & Model Training

API Usage

Example Request

Deployment

License

🧐 Overview

This repository hosts a FastAPI service that predicts loan default probabilities using a pre-trained machine learning model. The model is downloaded from Google Drive on startup to keep the GitHub repo lightweight.

🎬 Demo

Once running locally, view the interactive Swagger UI:🔗 http://127.0.0.1:8000/docs

🗂 Repository Structure

loan-default-api/
├── app.py                # FastAPI application & model loader
├── get_features.py       # Feature engineering helper (if used)
├── script.py             # Data extraction, preprocessing, & model training
├── requirements.txt      # Python dependencies
└── .gitignore            # Ignored files/folders

The models/ directory is created at runtime and not stored here.

⚙️ Setup

Clone the repo

git clone https://github.com/dasegovi19/loan-default-api.git
cd loan-default-api

Activate your environment

Conda:

conda activate loan_env

venv:

source venv/bin/activate

Install dependencies

pip install -r requirements.txt

🛠 Data Preparation & Model Training

Use script.py to:

Extract Loan_default.csv from the ZIP archive

Preprocess data with pandas

Train & evaluate models (Logistic Regression, Random Forest)

Save the best model to models/best_model.joblib

python script.py

🚀 API Usage

Start the server

uvicorn app:app --reload

Explore & test via Swagger UI:
http://127.0.0.1:8000/docs

💡 Example Request

Endpoint: POST /predictPayload:

{
  "Age": 35,
  "Income": 65000,
  "LoanAmount": 15000,
  "CreditScore": 680,
  "MonthsEmployed": 24,
  "NumCreditLines": 5,
  "InterestRate": 7.5,
  "LoanTerm": 36,
  "DTIRatio": 0.25,
  "DTI": 0.30,
  "Education_High School": 1,
  "Education_Master's": 0,
  "Education_PhD": 0,
  "EmploymentType_Part-time": 0,
  "EmploymentType_Self-employed": 0,
  "EmploymentType_Unemployed": 0,
  "MaritalStatus_Married": 1,
  "MaritalStatus_Single": 0,
  "HasMortgage_Yes": 1,
  "HasDependents_Yes": 0,
  "LoanPurpose_Home": 1,
  "FICO_Band_Medium": 1
}

Response:

{ "default_probability": 0.1623 }

☁️ Deployment

Recommend platforms:

Render.com: Auto-detects FastAPI

Deta: One-click deployment

Hugging Face Spaces: Use Dockerfile or space.toml

Ensure the deployment environment can fetch from Google Drive or switch to S3/Azure Blob as needed.

📄 License

This project is released under the MIT License.
