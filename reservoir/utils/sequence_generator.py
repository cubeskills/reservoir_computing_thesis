import numpy as np


def generate_ornstein_uhlenbeck(
    n_steps: int,
    theta: float = 0.7,
    mu: float = 0.0,
    sigma: float = 0.3,
    dt: float = 0.1,
    x0: float = 1.0,  
    seed: int = 42,

) -> np.ndarray:
    """Generate an Ornstein-Uhlenbeck process."""
    random_state = np.random.RandomState(seed=seed)
    x = np.zeros(n_steps)
    x[0] = mu if x0 is None else x0
    #sigma = np.sqrt(sigma)  
    #sigma = 0.3
    sigma = np.sqrt(theta/2) 

    sqrt_dt_sigma = np.sqrt(dt) * sigma
    for i in range(1,n_steps):
        # wiener process
        dx = theta * (mu - x[i - 1]) * dt + sqrt_dt_sigma * random_state.randn()
        x[i] = x[i - 1] + dx
    
    return x

def count_ones_in_window_task(n_steps: int, history_length: int = 5, input_dim: int = 1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:

    """Generate a task where the output is the parity of the sum of the last `history_length` bits."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_steps, input_dim))
    y = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        if t >= history_length - 1:
            window = X[t-history_length+1 : t+1, :]
            y[t] = int(window.sum())
        else:
            y[t] = 0

    return X.astype(float), y

def parity_task(
    n_steps: int,
    history_length: int = 3,
    input_dim: int = 1,
    seed: int = 42,
    task: str = "multi_sum",
) -> tuple[np.ndarray, np.ndarray]:
   
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_steps, input_dim))
    y = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        if t >= history_length - 1:
            window = X[t - history_length + 1 : t + 1, 0]  
            total_ones = window.sum()                       
            y[t] = int(total_ones % 2)
    
    return X.astype(float), y


def generate_mackey_glass(n_steps=10000, dim=1, x0=None, a=0.2, b=0.1, c=10.0, tau=23.0,
                 n=1000, sample=0.46, discard=250, seed=42) -> np.ndarray:
    
    main_rng = np.random.RandomState(seed)
    all_series = []

    for _ in range(dim):
        dim_seed = main_rng.randint(0, 2**32 - 1)
        dim_rng = np.random.RandomState(dim_seed)

        sample_interval = int(n * sample / tau)
        grids = n * discard + sample_interval * n_steps
        x = np.empty(grids)
        
        if x0 is None:
            x[:n] = 0.5 + 0.05 * (-1 + 2 * dim_rng.random(n))
        else:
            x[:n] = x0

        A = (2 * n - b * tau) / (2 * n + b * tau)
        B = a * tau / (2 * n + b * tau)

        for i in range(n - 1, grids - 1):
            x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                       x[i - n + 1] / (1 + x[i - n + 1] ** c))
        
        single_series = x[n * discard::sample_interval]
        
        if len(single_series) > n_steps:
            single_series = single_series[:n_steps]
        elif len(single_series) < n_steps:
            padding = np.zeros(n_steps - len(single_series))
            single_series = np.concatenate([single_series, padding])
        
        all_series.append(single_series)

    return np.stack(all_series, axis=1)




def random_process(n_steps: int = 5000,lower: float = -np.sqrt(3/4), upper: float = np.sqrt(3/4), seed: int = 42) -> np.ndarray:
    """Generate a random process."""
    rng = np.random.RandomState(seed)
    arr = rng.uniform(lower, upper, size=n_steps) 
    #print(arr)
    return arr.astype(float)  # Ensure the output is of type float
                   
def generate_narma(
    n_steps: int,
    dim: int = 1,
    order: int = 10,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 1.5,
    delta: float = 0.1,
    u_low: float = 0.0,
    u_high: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
   
    rng = np.random.RandomState(seed)
    u = rng.uniform(u_low, u_high, size=n_steps)
    y = np.zeros(n_steps)
    
    for t in range(order, n_steps - 1):
        past_y = y[t - order + 1 : t + 1] 
        y[t + 1] = (
            alpha * y[t]
            + beta * y[t] * np.sum(past_y)
            + gamma * u[t - order + 1] * u[t]
            + delta
        )
    
    return (u,y)