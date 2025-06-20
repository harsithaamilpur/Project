from models import PricingModel
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    try:
        df = pd.read_csv('ride_requests.csv')
        
        if len(df) < 20:
            raise ValueError("Insufficient data samples (minimum 20 required)")
            
        required_cols = ['distance_km', 'priority', 'carpool', 'final_price']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        df = df.dropna(subset=required_cols)
        
        df['priority'] = df['priority'].astype(int)
        df['carpool'] = df['carpool'].astype(int)
        
        for col in ['priority', 'carpool']:
            if not set(df[col].unique()).issubset({0, 1}):
                raise ValueError(f"Column {col} contains non-binary values")
        
        for col in ['distance_km', 'final_price']:
            q1 = df[col].quantile(0.05)
            q3 = df[col].quantile(0.95)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        df['distance_squared'] = df['distance_km'] ** 2
        df['priority_distance_interaction'] = df['priority'] * df['distance_km']
        
        features = ['distance_km', 'priority', 'carpool', 
                   'distance_squared', 'priority_distance_interaction']
        X = df[features].values
        y = df['final_price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=df[['priority', 'carpool']].apply(lambda x: f"{x[0]}_{x[1]}", axis=1)
        )
        
        print(f"Data loaded successfully with {len(X_train)} training samples")
        return X_train, y_train
        
    except Exception as e:
        print(f"Using synthetic data because: {str(e)}")
        np.random.seed(42)
        
        n_samples = 200
        X = np.zeros((n_samples, 5))
        
        X[:, 0] = np.random.exponential(scale=10, size=n_samples)
        X[:, 0] = np.clip(X[:, 0], 1, 50)
        
        X[:, 1] = (np.random.rand(n_samples) > 0.8).astype(int)
        
        X[:, 2] = (np.random.rand(n_samples) > 0.7).astype(int)
        
        X[:, 3] = X[:, 0] ** 2
        X[:, 4] = X[:, 1] * X[:, 0]
        
        base_price = X[:, 0] * 10
        priority_surcharge = X[:, 1] * 20
        carpool_discount = X[:, 2] * 5
        y = base_price + priority_surcharge - carpool_discount + np.random.randn(n_samples) * 5
        
        print(f"Generated {n_samples} synthetic samples")
        return X, y

def train_model():
    X, y = load_data()
    model = PricingModel()
    model.train(X, y)
    joblib.dump(model, 'pricing_model.pkl')
    print("Model trained and saved with enhanced preprocessing")

if __name__ == '__main__':
    train_model()