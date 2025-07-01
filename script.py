
## Load the data------------
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import zipfile
import pandas as pd
import os

# 1. Point to your zip file
zip_path = r"C:\Users\dsegovi2\Box\Loan\Loan_default.csv.zip"

# 2. Where to extract (same folder, or a subfolder)
extract_dir = r"C:\Users\dsegovi2\Box\Loan"

# 3. Extract all files
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)
print("✅ Extraction complete")

# 4. Build the CSV path (adjust name if it’s different)
csv_path = os.path.join(extract_dir, "Loan_default.csv")

# 5. Load into pandas
df = pd.read_csv(csv_path)

# 6. Inspect
print("Rows × cols:", df.shape)


print(df.head())


from sklearn.preprocessing import StandardScaler
import os

# Assuming df is already loaded
# 1. Drop any unused ID columns
df = df.drop(columns=['LoanID'])

# 2. Inspect missingness
print("Missing values per column:\n", df.isnull().sum())

# Debt-to-Income (if you want a fresh calc; otherwise skip)
df['DTI'] = df['LoanAmount'] / df['Income']

# FICO score bands
bins = [300, 580, 670, 740, 800, 850]
labels = ['VeryLow','Low','Medium','High','VeryHigh']
df['FICO_Band'] = pd.cut(df['CreditScore'], bins=bins, labels=labels)

print(df)

from sklearn.model_selection import train_test_split

# Assuming `df` is your fully prepped DataFrame and 'Default' is the target
#X = df.drop(columns=['Default'])
#y = df['Default']
# Drop the target for now
X = df.drop(columns=['Default'])

# Turn all categorical columns into numeric dummies
X = pd.get_dummies(X, drop_first=True)

# And pull the target back out
y = df['Default']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

y_pred_proba = lr.predict_proba(X_test)[:,1]
y_pred      = lr.predict(X_test)

print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba).round(4))
print(classification_report(y_test, y_pred, digits=4))




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Increase max_iter so the solver has time to converge
lr = LogisticRegression(
    max_iter=5000,           # up from 1000
    class_weight='balanced',
    random_state=42
)
lr.fit(X_train, y_train)

y_pred_proba = lr.predict_proba(X_test)[:, 1]
y_pred       = lr.predict(X_test)

print("Test AUC-ROC:", roc_auc_score(y_test, y_pred_proba).round(4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1. Initialize & train
rf = RandomForestClassifier(
    n_estimators=100,        # try 100 trees to start
    class_weight='balanced', # handle the imbalanced classes
    random_state=42
)
rf.fit(X_train, y_train)

# 2. Predict probabilities and classes
y_rf_proba = rf.predict_proba(X_test)[:, 1]
y_rf_pred  = rf.predict(X_test)

# 3. Evaluate
print("RandomForest AUC-ROC:", roc_auc_score(y_test, y_rf_proba).round(4))
print("\nClassification Report:\n", classification_report(y_test, y_rf_pred, digits=4))




