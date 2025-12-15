from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("loan_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        income = float(data["income"])
        loan_amount = float(data["loan_amount"])
        credit_score = float(data["credit_score"])
        employment_years = float(data["employment_years"])

        prediction = model.predict([[income, loan_amount, credit_score, employment_years]])
        probability = model.predict_proba([[income, loan_amount, credit_score, employment_years]])

        result = "Approved" if prediction[0] == 1 else "Rejected"
        approval_chance = round(probability[0][1] * 100, 2)

        return jsonify({
            "loan_status": result,
            "approval_probability": approval_chance
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run()
