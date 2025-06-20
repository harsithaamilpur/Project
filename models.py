import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer

class DQNModel:
    def __init__(self, state_size=7):  # Changed from 3 to 7 to match processed features
        self.state_size = state_size
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),  # Explicit input layer
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(0.001))
        return model
    
    def predict_optimal_price(self, state):
        return self.model.predict(np.array([state]))[0][0]

class PricingModel:
    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), [0]),  # distance_km
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), [1, 2])  # priority and carpool
            ],
            remainder='passthrough'
        )
        
        self.rf = RandomForestRegressor(n_estimators=100)
        self.dqn = DQNModel(state_size=7)  # Matches the 7 processed features
        
    def train(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)
        self.rf.fit(X_processed, y)
        
        rewards = y - (X[:,0] * 10)
        self.dqn.model.fit(X_processed, rewards, epochs=50, verbose=0)
        
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        rf_pred = self.rf.predict(X_processed)
        dqn_adjustment = np.array([self.dqn.predict_optimal_price(x) for x in X_processed])
        return (rf_pred + dqn_adjustment) / 2