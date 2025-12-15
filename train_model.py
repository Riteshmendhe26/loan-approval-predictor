import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Create dataset
data = {
    "income": [25000, 40000, 60000, 30000, 80000, 20000, 70000, 50000],
    "loan_amount": [200000, 250000, 300000, 180000, 400000, 150000, 350000, 280000],
    "credit_score": [650, 700, 750, 620, 800, 600, 770, 720],
    "employment_years": [2, 5, 8, 1, 10, 1, 7, 6],
    "loan_status": [0, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# 2. Split input and output
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Save model
joblib.dump(model, "loan_model.pkl")

print("Loan approval model trained and saved successfully!")
