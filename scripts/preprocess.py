from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    X = df.drop(columns=['default'])
    y = df['default']
    
    numeric_cols = X.select_dtypes(include='number').columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
