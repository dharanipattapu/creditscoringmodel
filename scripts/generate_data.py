import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 500

data = {
    'age': np.random.randint(18, 70, size=n_samples),
    'income': np.random.randint(20000, 120000, size=n_samples),
    'loan_amount': np.random.randint(1000, 50000, size=n_samples),
    'credit_score': np.random.randint(300, 850, size=n_samples),
}

# Generate probability of default
prob_default = (
    (850 - data['credit_score']) * 0.003 +
    (data['loan_amount'] / data['income']) * 0.4
)
data['default'] = (np.random.rand(n_samples) < prob_default).astype(int)

df = pd.DataFrame(data)
df.to_csv('data/credit_data.csv', index=False)
print("✅ Dataset created at data/credit_data.csv with 500 samples.")
