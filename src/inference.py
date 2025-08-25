import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn

from process import load_data  
from train import FraudDetectionModel  


def load_model(model_path, input_dim):
    model = FraudDetectionModel(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_single_transaction(model, scaler, transaction):
    transaction_scaled = scaler.transform([transaction])

    x_tensor = torch.tensor(transaction_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(x_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probability = torch.softmax(output, dim=1).numpy()[0]

    return predicted_class, probability


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_weights = load_data("data/creditcard.csv")

    input_dim = X_train.shape[1]

    scaler = StandardScaler().fit(X_train)

    model = load_model("fraud_model.pth", input_dim)

    sample = X_test[0]
    prediction, probability = predict_single_transaction(model, scaler, sample)

    print("Prediction (0=Not Fraud, 1=Fraud):", prediction)
    print("Probabilities:", probability)
