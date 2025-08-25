import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from process import load_data


class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  
        )

    def forward(self, x):
        return self.layers(x)  


def train_model():
    X_train, X_test, y_train, y_test, class_weights = load_data(r"C:/Users/ghada/Documents/ForInternships/fraudDetection/data/creditcard.csv")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    pos_weight = torch.tensor([class_weights[0] / class_weights[1]], dtype=torch.float32)

    model = FraudDetectionModel(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

   
    for epoch in range(10): 
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

 
    torch.save(model.state_dict(), "fraud_model.pth")
    print("✅ Model training complete and saved as fraud_model.pth")


    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits)  
        preds = (probs >= 0.5).float()

    y_true = y_test_tensor.numpy()
    y_pred = preds.numpy()
    y_prob = probs.numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n📊 Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")



if __name__ == "__main__":
    train_model()