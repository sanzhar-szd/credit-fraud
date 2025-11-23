# credit-fraud
Final project for AI essentials

This project explores credit card fraud detection on a highly imbalanced dataset using:

- Logistic Regression (baseline on original data)
- Logistic Regression with:
  - Random Undersampling (RUS)
  - SMOTE Oversampling
- A simple Neural Network (PyTorch) trained on SMOTE-resampled data
- Threshold tuning for the neural network to optimize the F1-score for the fraud class

All work is done in the Jupyter notebook: **`credit_card_fraud.ipynb`**.

The dataset can be downloaded via this link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The steps implemented:
1) EDA
2) Preprocessing
3) Train-test split
4) Handling class imbalance through random undersampling and oversampling
5) Logistic regression and simple neural network

Key Findings:
1) Baseline Logistic Regression (original data):
-  Very high overall accuracy (due to class imbalance)
-  Good precision for fraud, but relatively low recall (many fraudulent transactions are missed).
2) Logistic Regression with RUS/SMOTE:
-  Can significantly increase recall for fraud (catching most fraud cases)
-  Generates many false positives (many normal transactions flagged as fraud)
-  Leads to very low precision for fraud in some configurations.
3) Neural Network + SMOTE + threshold tuning:
Trained on a balanced (SMOTE) training set and evaluated on real imbalanced test set
With tuned threshold (≈ 0.95), achieves:
-  High recall for fraud
-  Much better precision compared to a naive 0.5 threshold
-  Strong F1-score for the minority class
-  Low false positive rate (more realistic for a production-like setting)

Overall, the project shows how:
Resampling (RUS/SMOTE),model choice (LogReg vs NN) and decision threshold tuning jointly affect performance on highly imbalanced fraud detection problems.
