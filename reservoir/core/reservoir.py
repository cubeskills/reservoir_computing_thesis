import numpy as np
from utils.utils import rescale_matrix
# reservoir class

class Reservoir:
    def __init__(self, 
                 input_dim: int, 
                 reservoir_size: int, 
                 activation_function: str = 'tanh', 
                 connectivity: float = 0.1, 
                 spectral_radius: float = 0.995, 
                 leak_rate: float = 1.0, 
                 scale: float = 1.0,
                 seed: int = 42):
        # standart parameters
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.scale = scale

        if activation_function == 'tanh':
            self.activation_function = np.tanh
        elif activation_function == 'relu':
            self.activation_function = lambda x: np.maximum(0, x)
        elif activation_function == 'sigmoid':
            self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        elif activation_function == 'linear':
            self.activation_function = lambda x: x
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
        
        
        # reproducible results
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        # Initialize weights
        self.W = None
        self.state = np.zeros(self.reservoir_size)
        self.current_history = []

        self.build()
       
    def build(self) -> None:
        """Build reservoir weights"""
        
        mask = self.random_state.rand(self.reservoir_size, self.reservoir_size) < self.connectivity
        #print(f"Mask shape: {mask.shape}, Mask sum: {np.sum(mask)} in build")
        W_raw = (self.random_state.rand(self.reservoir_size, self.reservoir_size) * 2 - 1) * mask 
        self.W = rescale_matrix(W_raw, self.spectral_radius)
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(self.W)))
        #print(f"Max eigenvalue: {max_eigenvalue}")
        #print(f"Spectral radius: {self.spectral_radius}")

        
    def step(self, X: np.ndarray) -> np.ndarray:

        """Udate reservoir state with PROJECTED input -> should already be in reservoir space"""
        # flatten if necessary
        X = np.asarray(X).flatten()
        if X.shape[0] != self.reservoir_size:
            raise ValueError(
                f"Projected input dimension {X.shape[0]} does not match "
                f"reservoir size {self.reservoir_size}"
            )
        
        pre = self.W @ self.state + X
        activated = self.activation_function(pre)
        new_state = (1 - self.leak_rate) * self.state + self.leak_rate * activated
        

        self.current_history.append(pre.copy())

        self.state = new_state
        return self.state
    
    def get_currents(self) -> np.ndarray:
        """Return np array of current reservoir states"""
        if not self.current_history:
            return np.empty((0, self.reservoir_size))
        return np.vstack(self.current_history)

    
    def reset_state(self, state: np.ndarray = None) -> None:
        """Reset the reservoir state to a given state or to zero if no state is provided"""
        if state is None:
            self.state = np.zeros((self.reservoir_size,))
        else:
            if state.shape[0] != self.reservoir_size:
                raise ValueError(f"State dimension {state.shape[0]} does not match reservoir size {self.reservoir_size}")
            self.state = np.copy(state)

    def reset_history(self) -> None:
        """Reset the history of the reservoir"""
        self.current_history.clear()
    
    def get_states(self) -> np.ndarray:
        """Get the current state of the reservoir"""
        return np.copy(self.state)
    
    
    
    
