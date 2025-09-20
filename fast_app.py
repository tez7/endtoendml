import pickle
from fastapi import FastAPI
import numpy as np

app = FastAPI()

# Load the model and scaler
model = pickle.load(open('dtmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.post("/predict_api")
async def predict_api(data: dict):
    # Process the data
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    
    # Make a prediction
    output = model.predict(new_data)
    
    # Return the prediction as a JSON response
    return {"prediction": int(output[0])}