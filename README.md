# Credit Card Fraud Detection

A Python project that detects fraudulent credit card transactions using PyTorch, NumPy, and Pandas.

---

## 🔹 Project Overview
- Uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
- Preprocesses the data (scaling features, handling class imbalance).  
- Trains a neural network to classify transactions as **Fraud (1)** or **Not Fraud (0)**.  
- Supports both **training** and **inference** for new transactions.  
- Evaluation metrics include **Accuracy**, **Precision**, **Recall**, and **F1-score**.

---

## 🔹 Installation

1. Clone the repo:

```bash
git clone https://github.com/gghada/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place creditcard.csv in the data/ folder.

## 🔹 Usage 

1.  Train the Model
```bash
python src/train.py
```

2. Run Inference
```bash
python src/inference.py
```
