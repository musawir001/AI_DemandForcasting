import joblib

def load_artifacts():
    model = joblib.load("demand_forecasting_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    features = joblib.load("features.pkl")

    return model, label_encoder, features