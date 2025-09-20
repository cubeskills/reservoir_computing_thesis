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

spectral_radii = np.linspace(0.2, 1.8, 20)
#scales = np.array([0.1,0.5, 1.0])
scales = [0.1,0.5,1.0]
seed = 42
random_process_sets = [
    compute_random_processes(1, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(3, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(5, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(7, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),

]
X_sets = [p.T for p in random_process_sets]

TASKS = ["delay"]
trials = 10
def main():
    total_runs = len(scales) * len(spectral_radii) * len(X_sets) * len(TASKS) * trials
    results = []

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
                                sparse_mapping=False,
                            )
                            
                            X, y = prep_task(task, X_full, history_length=10,act=True)
                            n_train = int(len(y) * 0.8)
                            X_train, y_train = X[:n_train], y[:n_train]
                            X_test, y_test = X[n_train:], y[n_train:]
                            
                            trainer.train(X_train,y_train)
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
                                    process_count = X.shape[1],
                                    task = task,
                                    spectral_radius = rho,
                                    scale = scale,
                                    participation_ratio = pr,
                                    variance_of_activation_derivatives = vad,
                                    average_state_entropy = ase,
                                    active_information_storage = ais,
                                    mean_correlation = mean_corr,
                                    transfer_entropy = te,
                                )
                            )
                            pbar.update(1)
        
    df = pd.DataFrame(results)
    df.to_csv("../THESIS_CSV/ALL_MEASURES_RP.csv", index=False)
    print("Results saved to ../THESIS_CSV/ALL_MEASURES_RP.csv")


if __name__ == "__main__":
    main()