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

random_process_sets = [
    compute_random_processes(1, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(3, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(5, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(7, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
]

#process_sets = ornstein_uhlenbeck_sets + random_process_sets
X_sets = [p.T for p in random_process_sets]
#names = ["OU_5","OU_15","RP_5", "RP_15"]
TASKS = ["delay", "delay_polynomial", "polynomial","narma"]
trials = 10
def main():
    results = []
    total_runs = trials * len(X_sets) * len(TASKS) * len(scales) * len(spectral_radii)
    with tqdm(total=total_runs, desc="ESN measure sweep", unit="run") as pbar:
        for trial in range(1,trials+1):
            for X_full in X_sets:
                for scale in scales:
                    for rho in spectral_radii:

                        for task in TASKS:
                            n_inputs = X_full.shape[1]
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
                            
                            X, y = prep_task(task, X_full, history_length=10,act=True)
                            n_train = int(len(y) * 0.8)
                            X_train, y_train = X[:n_train], y[:n_train]
                            X_test, y_test = X[n_train:], y[n_train:]
                            
                            trainer.train(X_train,y_train)
                            y_pred = trainer.predict(X_test)
                            nmse_error = nmse(y_test, y_pred)
                            results.append(
                                dict(
                                    trial = trial,
                                    process_count = X.shape[1],
                                    task = task,  
                                    scale = scale,
                                    spectral_radius= rho,
                                    nmse_error = nmse_error,

                            
                                )
                            )
                            pbar.update(1)

    


        df = pd.DataFrame(results)
        df.to_csv("../THESIS_CSV/ALL_TASKS_NMSE_OU_THESIS_SPARSE.csv", index=False)

if __name__ == "__main__":
    main()
