import numpy as np

class InputMapping:
    def __init__(self, input_dim: int, reservoir_size: int, scale: float = 1.0,seed: int = 42, sparse_mapping: bool = False):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        # scale quantifies the strenth of input signal (proportional to leak rate?)
        self.sparse_mapping = sparse_mapping
        self.scale = scale
        self.random_state = np.random.RandomState(seed=seed)
        self.weights = self.generate_weights()
        
    def generate_weights(self) -> np.ndarray:
        """ Generate input weights"""

        weights = self.random_state.rand(self.reservoir_size, self.input_dim) * 2 - 1
        if self.sparse_mapping:
            # make the mapping sparse
            mask = self.random_state.rand(*weights.shape) < 0.1
            weights *= mask
            
        return weights * self.scale
    
    def map_input(self, input: np.ndarray) -> np.ndarray:
        """ Map input to reservoir space"""
        # resulting shape: (inputs.shape[0], self.reservoir_size)
        input_signal = np.asarray(input).reshape(-1)

        if input_signal.shape[0] != self.input_dim:
            raise ValueError(f"Input size {input_signal.shape[0]} does not match expected size {self.input_dim}.")
        
        return self.weights @ input_signal
    
    def reset(self, new_seed=None):
        """Reset the input mapping with a new random seed"""
        if new_seed is not None:
            self.random_state = np.random.RandomState(seed=new_seed)
        self.weights = self.generate_weights()