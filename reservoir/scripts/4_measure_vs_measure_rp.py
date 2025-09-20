import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from utils.simulation_tools import compute_random_processes, build_reservoir, prep_task
from utils.sequence_generator import parity_task, count_ones_in_window_task
from utils.metrics import mse, nmse
from utils.measures import Measures

seed = 42
theta = 0.7

random_set = compute_random_processes(
    n_samples=1, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed
)

if isinstance(random_set, list):
    X_full = random_set[0]
else:
    X_full = random_set

if X_full.ndim == 1:
    X_full = X_full.reshape(-1, 1)
elif X_full.ndim == 2 and X_full.shape[0] == 1:
    X_full = X_full.T

X_sets = [X_full]

TASKS = ["delay"]
trials = 10
spectral_radii = np.linspace(0.2, 1.8, 60)
scale = 0.1

def main():
    results = []
    total_runs = trials * len(TASKS) * len(spectral_radii)
    
    with tqdm(total=total_runs, desc="ESN measure sweep", unit="run") as pbar:
        for trial in range(1, trials + 1):
            for sr in spectral_radii:
                for task in TASKS:
                    X_input = X_sets[0].copy()
                    n_inputs = X_input.shape[1]
                    
                    X, y = prep_task(task, X_input, history_length=10, act=True, seed=trial)
                    
                    n_train = int(len(y) * 0.8)
                    X_train, y_train = X[:n_train], y[:n_train]
                    X_test, y_test = X[n_train:], y[n_train:]
                    
                    trainer = build_reservoir(
                        input_dim=n_inputs,
                        reservoir_size=100,
                        spectral_radius=sr,
                        connectivity=0.1,
                        leak_rate=1.0,
                        scale=scale,
                        seed=trial,
                        sparse_mapping=False,
                    )
                    
                    trainer.train(X_train, y_train)
                    y_pred = trainer.predict(X_test)
                    nmse_error = nmse(y_test, y_pred)
                    
                    meas = Measures(trainer.current_history, trainer.states)
                    pr = meas.participation_ratio()
                    vad = meas.variance_of_activation_derivatives()
                    ase = meas.average_state_entropy()
                    ais = meas.ais(k=1)
                    mean_corr = meas.mean_correlation()
                    te = meas.te(k=1)
                    
                    results.append(
                        dict(
                            trial=trial,
                            process_count=X.shape[1],
                            process="random_process",
                            spectral_radius=sr,
                            scale=scale,
                            nmse=nmse_error,
                            participation_ratio=pr,
                            variance_of_activation_derivatives=vad,
                            average_state_entropy=ase,
                            active_information_storage=ais,
                            mean_correlation=mean_corr,
                            transfer_entropy=te
                        )
                    )
                    pbar.update(1)
    
    df = pd.DataFrame(results)
    df.to_csv("../THESIS_CSV/ALL_MEASURES_CORR_RP.csv", index=False)

if __name__ == "__main__":
    main()