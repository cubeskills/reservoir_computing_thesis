import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from utils.simulation_tools import build_reservoir, prep_task, compute_random_processes
from utils.metrics import mse,nmse
from utils.measures import Measures
from utils.sequence_generator import parity_task,count_ones_in_window_task
spectral_radii = np.linspace(0.2,1.8, 20)
#scales = np.array([0.1,0.5, 1.0])
scales = [0.1,0.5,1.0]
seed = 42
theta = 0.7  # keep variance at 1/4


#names = ["OU_5","OU_15","RP_5", "RP_15"]
BINARY_TASKS = ["parity","counting_task"]
trials = 10
def main():
    results = []
    total_runs = trials * 4 * len(BINARY_TASKS) * len(scales) * len(spectral_radii)
    with tqdm(total=total_runs, desc="ESN measure sweep", unit="run") as pbar:
        
        for trial in range(1,trials+1):
            for input_dim in [1,3,5,7]:
                for task in BINARY_TASKS:
                    if task == "parity":
                        X,y = parity_task(n_steps = 5000,
                                        history_length=2,
                                        input_dim=input_dim,
                                        seed=trial,
                                        task="multi_sum")
                    if task == "counting_task":
                        X,y = count_ones_in_window_task(n_steps=5000,
                                                        history_length=5,
                                                        input_dim=input_dim,
                                                        seed=trial)
                    n_inputs = X.shape[1]
                    X, y = X.astype(float), y.astype(float)
                    #print(f"y[:20]: {y[:20]}")
                    n_train = int(len(y) * 0.8)
                    X_train, y_train = X[:n_train], y[:n_train]
                    X_test, y_test = X[n_train:], y[n_train:]

                    for scale in scales:
                        for sr in spectral_radii:
                            rho = sr
                        
                            
                            trainer = build_reservoir(
                                input_dim=n_inputs,
                                reservoir_size=100,
                                spectral_radius=rho,
                                connectivity=0.1,
                                leak_rate=1.0,
                                scale=scale,
                                seed=trial,
                                sparse_mapping=True,
                            )

                            trainer.train(X_train,y_train)
                            y_pred = trainer.predict(X_test)
                            mse_error = mse(y_test,y_pred)
                            nmse_error = nmse(y_test,y_pred)
                            results.append(
                                dict(
                                    trial = trial,
                                    process_count = X.shape[1],
                                    task = task, 
                                    scale = scale,
                                    spectral_radius= rho,
                                    nmse_error = nmse_error,
                                    mse_error = mse_error,
                                )
                            )
                            pbar.update(1)


        df = pd.DataFrame(results)
        df.to_csv("../THESIS_CSV/ALL_TASKS_NMSE_BINARY_THESIS_SPARSE.csv", index=False)

if __name__ == "__main__":
    main()
