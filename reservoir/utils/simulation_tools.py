import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.reservoir import Reservoir
from core.readout import RidgeReadout 
from core.input_mapping import InputMapping
from training.trainer import ESNTrainer
from utils.sequence_generator import generate_ornstein_uhlenbeck, random_process, generate_narma, generate_mackey_glass


def compute_ou_processes(n_samples: int=5, n_total_steps: int = 5000, seed: int = 42, theta=0.7) -> np.ndarray:
    """
    Generate n_samples Ornstein-Uhlenbeck processes.
    """
    ou_processes = []
    for i in range(n_samples):
        ou_process = generate_ornstein_uhlenbeck(n_total_steps,theta=theta, seed=seed + i)
        ou_processes.append(ou_process)
    
    return np.array(ou_processes)

def compute_random_processes(n_samples: int=5, n_total_steps: int = 5000, lower: float = -np.sqrt(3/4), upper: float = np.sqrt(3/4), seed: int = 42) -> np.ndarray:
    """ Generate n_samples random processes."""
    rps = []
    for i in range(n_samples):
        rp = random_process(n_steps=n_total_steps, lower=lower, upper=upper, seed=seed + i)
        #print(rp)
        rps.append(rp)
    return np.array(rps)

def compute_narma_processes(n_samples: int=5, n_total_steps: int = 5000, seed: int = 42) -> np.ndarray:
    """
    Generate n_samples NARMA processes.
    """
    narma_processes = []
    for i in range(n_samples):
        narma_process = generate_narma(n_total_steps, seed=seed + i)
        narma_processes.append(narma_process)
    
    return np.array(narma_processes)

def compute_mackey_glass_processes(n_samples: int=5, n_total_steps: int = 5000, seed: int = 42) -> np.ndarray:

    """
    Generate n_samples Mackey-Glass processes.
    """
    mackey_glass_processes = []
    for i in range(n_samples):
        mackey_glass_process = generate_mackey_glass(n_total_steps, seed=seed + i)
        mackey_glass_processes.append(mackey_glass_process)
    
    return np.array(mackey_glass_processes)

def build_reservoir(
    input_dim: int,
    reservoir_size: int = 100,
    spectral_radius: float = 0.95,
    connectivity: float = 0.1,
    leak_rate: float = 0.9,
    scale: float = 1,
    seed: int = 42,
    sparse_mapping: bool = False
) -> ESNTrainer:
    
    mapping = InputMapping(input_dim=input_dim, reservoir_size=reservoir_size, scale=scale, seed=seed, sparse_mapping=sparse_mapping)

    reservoir = Reservoir(
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        connectivity=connectivity,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        seed=seed,
    )
    
    readout = RidgeReadout(output_dim=1, reservoir_size=reservoir_size)
    
    return ESNTrainer(reservoir, readout, mapping)

def prep_task(task: str, X_full: np.ndarray,
              history_length: int = 50,
              act= False,
              seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
   
    T, N = X_full.shape
    rng = np.random.default_rng(seed)

    if task == "reconstruction":

        X_task = X_full.copy()
        y_task = X_task[:, [0]].copy()
        if act:
            y_task = np.tanh(y_task)
        return X_task, y_task

    if task == "sum":
        X_task = X_full.copy()
        y_task = X_task.sum(axis=1, keepdims=True).copy()
        if act:
            y_task = np.tanh(y_task)
        return X_task, y_task

    if task == "polynomial":
        X_task = X_full.copy()

        poly = X_full[:, 0].copy()      
        count = 3
        for i in range(1, N):
            poly = poly + (X_full[:, i] ** count)
            count = (count + 2) % 10

        y_task = poly.reshape(-1, 1).copy()
        if act:
            y_task_2 = np.tanh(y_task.copy())
        else:
            y_task_2 = y_task.copy()
        return X_task, y_task_2

    if task == "delay":
        d = history_length
        X_task = X_full[d:].copy()
        y_task = X_full[:-d, [0]].copy()
        if act:
            y_task = np.tanh(y_task)
        return X_task, y_task
    
    if task == "delay_polynomial":
        if N > 7:
            delays = rng.integers(0, history_length, size=N)
        else:
            delays = np.array([10,7,5,3,10,7,5,3])
            delays = delays[:N]
        #print(delays)
        powers = [(1 + 2*j) % 10 for j in range(N)]
        #print(powers)
        max_delay = int(delays.max())
        T2 = T - max_delay

        X_task = X_full[max_delay:].copy()

        y = np.zeros((T2,), dtype=float)
        for t in range(T2):
            acc = 0.0
            for j in range(N):
                idx = t + max_delay - delays[j]
                acc += (X_full[idx, j] ** powers[j])
            y[t] = acc

        y_task = y.reshape(-1, 1).copy()
        #print(y_task[:20])
        if act:
            y_task_2 = np.tanh(y_task.copy())
        else:
            y_task_2 = y_task.copy()
        return X_task, y_task_2
    
    if task == "narma":
        order = 10
        alpha = 0.2
        beta = 0.004
        gamma = 1.5
        delta = 0.001

        T = X_full.shape[0]
        
        u = X_full[:, 0].copy()
        # map u to [0,0.5]
        u = (u - u.min()) / (u.max() - u.min()) * 0.5
        y_temp = np.zeros(T, dtype=float)
        for t in range(order - 1, T - 1):
            y_sum = np.sum(y_temp[t - (order - 1) : t + 1])
            
            y_temp[t+1] = (alpha * y_temp[t] + 
                        beta * y_temp[t] * y_sum + 
                        gamma * u[t - (order - 1)] * u[t] + 
                        delta)

        X_task = X_full[order:].copy()
        y_task = y_temp[order:].reshape(-1, 1).copy()
        if act:
            y_task_2 = np.tanh(y_task.copy())
        else:
            y_task_2 = y_task.copy()
        return X_task, y_task_2
       
    raise ValueError(f"Unknown task: {task!r}")
