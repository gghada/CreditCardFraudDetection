import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Dataset shape:", data.shape)
    print("Data columns:", data.columns.tolist())
    print(data.head)

    x = data.drop('Class', axis=1)
    y = data['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {cls: weight for cls, weight in zip(classes, weights)}
    print("Class weights", class_weights)

    return X_train, X_test, y_train, y_test, class_weights

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_weights = load_data("../data/creditcard.csv")
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)