import streamlit as st
import math

# Apply some custom CSS for a pink-to-blue gradient background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #ffc0cb, #87cefa);
    }
    .reportview-container .main {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and header
st.title("Veronica Project")
st.header("Enter Data to Predict Default Probability")

# Input fields with limited ranges
annual_income = st.number_input(
    "Annual Income", min_value=1000.0, max_value=1000000.0, value=50000.0, step=1000.0
)
credit_score = st.number_input(
    "Credit Score", min_value=300.0, max_value=850.0, value=600.0, step=1.0
)

# Define the simple logistic model class
class SimpleLogisticModel:
    def __init__(self):
        # Adjusted weights and bias for a more dynamic prediction
        self.w1 = 1e-5   # Weight for annual_income
        self.w2 = 0.03   # Weight for credit_score
        self.b  = -4.0   # Bias term

    def predict_proba(self, inputs):
        """
        inputs: list of [annual_income, credit_score]
        returns: [[prob_not_default, prob_default], ...]
        """
        results = []
        for annual_income, credit_score in inputs:
            z = (self.w1 * annual_income) + (self.w2 * credit_score) + self.b
            p_default = 1.0 / (1.0 + math.exp(-z))
            p_not_default = 1.0 - p_default
            results.append([p_not_default, p_default])
        return results

# Instantiate the model
model = SimpleLogisticModel()

# When the user clicks "Predict", calculate and display the result
if st.button("Predict"):
    prediction_proba = model.predict_proba([[annual_income, credit_score]])[0][1]
    st.success(f"Probability of Default: {prediction_proba * 100:.2f}%")
    st.info(
        "Computed using a logistic function:\n"
        "p = 1 / (1 + e^(-z))\n"
        "where z = (w1 * annual_income) + (w2 * credit_score) + b"
    )
    # In Streamlit, inputs are re-rendered on each run, so the form is effectively 'reset'
