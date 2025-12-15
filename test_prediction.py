import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

# New loan applicant data
# income, loan_amount, credit_score, employment_years
new_applicant = [[50000, 250000, 720, 5]]

# Predict approval
prediction = model.predict(new_applicant)

print("Prediction:", prediction)
