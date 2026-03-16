# ============================================
# REFINED CREDIT DELINQUENCY MODEL TRAINING
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_excel("dataset/Delinquency_prediction_dataset.xlsx")

print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)
print("\nMissing values before cleaning:\n", data.isnull().sum())

# -----------------------------
# 2. Encode payment statuses
# -----------------------------
status_map = {'On-time': 0, 'Late': 1, 'Missed': 2}
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
for col in month_cols:
    if col in data.columns:
        data[col] = data[col].map(status_map)

# -----------------------------
# 3. Feature engineering
# -----------------------------
data['Total_Missed'] = data[month_cols].apply(lambda x: (x == 2).sum(), axis=1)
data['Total_Late'] = data[month_cols].apply(lambda x: (x == 1).sum(), axis=1)

# -----------------------------
# 4. Handle missing values
# -----------------------------
# Fill numeric NaNs with median
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill categorical NaNs with mode
categorical_cols = data.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:\n", data.isnull().sum())

# -----------------------------
# 5. Define features and target
# -----------------------------
target = 'Delinquent_Account'
X = data.drop(columns=['Customer_ID', target])
y = data[target]

# -----------------------------
# 6. Preprocessing pipeline
# -----------------------------
categorical_features = ['Employment_Status', 'Credit_Card_Type', 'Location']
numeric_features = [col for col in X.columns if col not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# -----------------------------
# 7. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n✅ Data split complete.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------
# 8. Model training
# -----------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("\n🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Model training complete!")

# -----------------------------
# 9. Save model
# -----------------------------
joblib.dump(model, "saved_model/model.pkl")
print("💾 Model saved successfully at saved_model/model.pkl")