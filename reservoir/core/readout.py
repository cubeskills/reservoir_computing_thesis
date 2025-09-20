import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

class Readout:
    # This is the base class, the actual code is in linear and ridge readout classes
    def __init__(self, output_dim: int, reservoir_size: int):
        self.output_dim = output_dim
        self.reservoir_size = reservoir_size
        self.W_out = None
        self.is_trained = False
    
    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """fit base class"""
        raise NotImplementedError("fit() is implemented in LinearReadout and RidgeReadout classes.")

    def predict(self, states: np.ndarray) -> np.ndarray:
        """predict base class"""
        if not self.is_trained:
            raise ValueError("Readout layer is not trained yet. Call fit() first.")
        return np.dot(states, self.W_out.T)
    

class LinearReadout(Readout):
    def __init__(self, output_dim: int, reservoir_size: int):
        # just linear regression
    
        super().__init__(output_dim, reservoir_size)
        self.model = LinearRegression()

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """fit model with linear regression"""
        self.model.fit(states, targets)
        self.W_out = self.model.coef_
        self.is_trained = True
        return self
    
    def predict(self, states: np.ndarray) -> np.ndarray:
        """predict with linear regression"""
        
        if not self.is_trained:
            raise ValueError("Readout layer is not trained. ")
        return self.model.predict(states)


class RidgeReadout(Readout):
    def __init__(self, output_dim: int, reservoir_size: int, alpha: float = 1e-4):
        
        super().__init__(output_dim, reservoir_size)
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """fit model with ridge"""
        self.model.fit(states, targets)
        self.W_out = self.model.coef_
        self.is_trained = True
        return self
    
    def predict(self, states: np.ndarray) -> np.ndarray:
        """predict with ridge"""
        if not self.is_trained:
            raise ValueError("Readout layer is not trained. ")
        return self.model.predict(states)

