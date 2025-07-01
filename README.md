Loan Default Prediction API

Overview

This repository contains a FastAPI service that predicts loan default probabilities using a pre-trained scikit-learn model. It also includes data extraction and model training scripts for reference and experimentation.

Repository Structure

app.py: FastAPI application defining the /predict endpoint and model-loading logic (downloads model from Google Drive at startup).

get_features.py: Helper module for feature processing (if applicable).

script.py: Data loading, preprocessing, and model training script demonstrating logistic regression and random forest workflows.

models/: Directory where the model file (best_model.joblib) is downloaded at runtime (not stored in Git).

requirements.txt: List of Python dependencies.

.gitignore: Specifies files and directories ignored by Git.

README.md: Project overview, setup, and usage instructions.

Setup

Clone the repository

git clone https://github.com/dasegovi19/loan-default-api.git
cd loan-default-api

Activate your Python environment
If using Conda:

conda activate loan_env

Or for a venv:

source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Data Preparation & Model Training

The script.py file contains a complete example of:

Extracting Loan_default.csv from a ZIP archive.

Loading and preprocessing the data with pandas.

Training and evaluating baseline models (Logistic Regression, Random Forest).

Saving the trained model to the models/ directory as best_model.joblib.

To run the script and train models locally:

python script.py

API Usage

Start the FastAPI server

uvicorn app:app --reload

Interactive API docs
Open your browser to:

http://127.0.0.1:8000/docs

This Swagger UI lets you explore and test the /predict endpoint.

Example /predict Request

JSON payload:

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
  "LoanPurpose_Business": 0,
  "LoanPurpose_Education": 0,
  "LoanPurpose_Home": 1,
  "LoanPurpose_Other": 0,
  "HasCoSigner_Yes": 0,
  "FICO_Band_Low": 0,
  "FICO_Band_Medium": 1,
  "FICO_Band_High": 0,
  "FICO_Band_VeryHigh": 0
}

Sample response:

{
  "default_probability": 0.1623
}

Deployment

To deploy this API in production, consider platforms like Render.com, Deta, or Hugging Face Spaces. Ensure that your chosen environment has network access to Google Drive to fetch the model or substitute with another storage solution (S3, Azure Blob, etc.).

License

This project is licensed under the MIT License. Feel free to use and modify it for your own portfolio or production needs.
