import pandas as pd
import os
import sys

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your preprocess function from scripts folder
from scripts.preprocess import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Build absolute path to data file (safe regardless of current working dir)
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'credit_data.csv'))

# Load data
df = pd.read_csv(data_path)

# Preprocess and split
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
