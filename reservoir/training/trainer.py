import numpy as np
from core.reservoir import Reservoir
from core.readout import Readout
from core.input_mapping import InputMapping
# import tuple
#from typing import Tuple
import time
from typing import Optional, Sequence

class ReservoirTrainer:
    def __init__(self, reservoir: Reservoir, readout: Readout, input_mapping: InputMapping = None):
        self.reservoir = reservoir
        self.readout = readout
        self.input_mapping = input_mapping
        self.is_trained = False
        self.training_time = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """implemented by subclasses"""
        raise NotImplementedError("train() is implemented in subclasses.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """implemented by subclasses"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        raise NotImplementedError("predict() is implemented in subclasses.")
    
    def reset(self) -> None:
        self.is_trained = False
        self.training_time = None
        self.reservoir.reset_state()
        
    def collect_states(self, X: np.ndarray) -> np.ndarray:
        """implemented by subclasses"""
        raise NotImplementedError("collect_states() is implemented in subclasses.")

class ESNTrainer(ReservoirTrainer):
    def __init__(self, reservoir: Reservoir, readout: Readout, input_mapping: InputMapping = None, auto_washout: bool = False, washout: int = 200):
        super().__init__(reservoir, readout, input_mapping)
        self.states = None
        self.washout = washout
        self.auto_washout = auto_washout
        self.current_history = []
        
       
    def collect_states(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Return all the states of reservoir for given input"""
        n_samples = X.shape[0]
        #if reset:
        #    self.reservoir.reset_state() 
        
        collected_states = np.zeros((n_samples, self.reservoir.reservoir_size))
        # iterate through samples 
        for i in range(n_samples):
            x_i = X[i].flatten()
            if self.input_mapping:
                x_i = self.input_mapping.map_input(x_i)
            self.reservoir.step(x_i) 
            collected_states[i] = self.reservoir.get_states() 
            
        self.current_history = self.reservoir.get_currents()
            
        return collected_states
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Train readout layer with ESN """
        # collect states
        start_time = time.time()
        X = np.array(X)
        y = np.array(y)
    
        self.states = self.collect_states(X)
        if self.auto_washout:
            self.washout = int(self.correlation_time(X)*5)

        states_after_washout = self.states[self.washout:]
        y_after_washout = y[self.washout:]

        self.readout.fit(states_after_washout, y_after_washout)

        self.is_trained = True 
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        prediction_states = self.collect_states(X, reset=False)
        predictions = self.readout.predict(prediction_states)
        return predictions
    
    def correlation_time(self, X: np.ndarray) -> float:
        """Calculate the correlation time of the reservoir states"""
        if self.current_history is None:
            self.collect_states(X)

        H = self.current_history  
        T, N = H.shape
        auto_correlations = []
        H_centered = H - H.mean(axis=0)
        var = (H_centered**2).mean(axis=0)
        for dt in range(1, T//2):
            corr = (H_centered[:-dt] * H_centered[dt:]).mean(axis=0) / var
            auto_correlations.append(corr.mean())
        auto_correlations = np.array(auto_correlations)
        # find first dt where acf < 1/e
        below = np.where(auto_correlations < 1/np.e)[0]
        return float(below[0]) if below.size > 0 else float(T//2)

    def compute_ntc(self,ranges: Sequence[float], corr_lengths: Sequence[float], C: Optional[float] = None) -> float:
        """Compute the NTC look in gao et al.
        call like this:
        trainer.train(X, y)
        y_pred = trainer.predict(X)

        T_sec = X.shape[0] * dt
        tau_est = trainer.correlation_time(X)
        
        ntc = trainer.compute_ntc(
        ranges=[T_sec],
        corr_lengths=[tau_est * dt]   # convert tau from steps → seconds
        )   
        """
        L = np.array(ranges, dtype=float)
        lamb = max(np.array(corr_lengths, dtype=float), 1e-10)
        K = L.size
        if C is None:
            C = (np.sqrt(2/np.pi))**K
        return C * np.prod(L / lamb)
    


    def generate_sequence(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """Generate a sequence of states from the reservoir"""
        pass


    