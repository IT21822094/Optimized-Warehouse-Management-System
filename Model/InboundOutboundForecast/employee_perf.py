import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

def predict_performance(input_data_dict):
    model = joblib.load(os.path.join(BASE_DIR, "warehouse_model.pkl"))
    shift_encoder = joblib.load(os.path.join(BASE_DIR, "shift_encoder.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    input_df = pd.DataFrame([input_data_dict])
    input_df['shift'] = shift_encoder.transform(input_df['shift'])
    prediction = model.predict(input_df)
    return label_encoder.inverse_transform(prediction)[0]
