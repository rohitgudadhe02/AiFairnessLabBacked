from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# This allows your Javascript web app to communicate with this Python API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model we just trained!
model = joblib.load('loan_approval_model.pkl')


# We need to tell the API exactly what data the web app will send.
# Note: You might need to adjust these names to exactly match your dataset columns
# (the 11 features that were printed out earlier, e.g., no_of_dependents, education, bank_asset_value)
class LoanApplication(BaseModel):
    no_of_dependents: float
    education: float
    self_employed: float
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.post("/predict")
def predict_approval(data: LoanApplication):
    # Convert the incoming web data into a Pandas DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Get probability of approval
    # predict_proba returns something like [[0.2, 0.8]] (20% reject, 80% approve)
    # So we grab the second number [0][1]
    probability = model.predict_proba(input_data)[0][1]

    # We also get the hard yes/no prediction
    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "approval_probability": float(probability)
    }
