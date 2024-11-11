from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = joblib.load(r'/model.pkl')

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to process form data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = request.form.get('Age')
        income = request.form.get('Annual_Income')
        credit_score = request.form.get('Credit_Score')
        employment_years = request.form.get('Employment_Years')
        loan_amount = request.form.get('Loan_Amount_Requested')
        
        # Convert form data to numpy array and reshape
        input_features = np.array([[age, income, credit_score, employment_years, loan_amount]], dtype=float)
        
        # Append placeholder values for the remaining features
        # Replace '0' with appropriate values as needed
        input_features = np.append(input_features, [0] * (28 - 5)).reshape(1, -1)

        # Predict default probability
        prediction = model.predict(input_features)
        result = "Default" if prediction[0] == 1 else "No Default"
        
        # Display result on result page
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {e}", 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
