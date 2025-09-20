import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from utils.simulation_tools import compute_random_processes, build_reservoir, prep_task
from utils.sequence_generator import parity_task, count_ones_in_window_task
from utils.metrics import mse,nmse
from utils.measures import Measures
seed=42


TASKS = ["delay","delay_polynomial","polynomial","narma","parity","counting_task"]
random_process_sets = [
    compute_random_processes(1, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(3, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(5, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
    compute_random_processes(7, n_total_steps=5000, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed),
]
spectral_radii = np.linspace(0.2,1.8, 30)
X_sets = [p.T for p in random_process_sets]
#scales = np.array([0.1,0.5, 1.0])
scales = [0.1,0.5,1.0]
trials = 10
def main():
    
    results = []
    total_runs = len(scales) * len(TASKS) * len(spectral_radii) * trials * len(X_sets)
    with tqdm(total=total_runs, desc="ESN measure sweep", unit="run") as pbar:
        for trial in range(trials):
            for X_full in X_sets:
                
                for scale in scales:    
                    for sr in spectral_radii:

                        n_inputs = X_full.shape[1]
                        trainer_meas = build_reservoir(
                            input_dim=n_inputs,
                            reservoir_size=100,
                            spectral_radius=sr,
                            connectivity=0.1,
                            leak_rate=1.0,
                            scale=scale,
                            seed=trial,
                            sparse_mapping=False,
                        )
                        states = trainer_meas.collect_states(X_full)
                        current_history = trainer_meas.current_history
                        
                        meas = Measures(current_history, states)
                        pr = meas.participation_ratio()
                        vad = meas.variance_of_activation_derivatives()
                        ase = meas.average_state_entropy()
                        ais = meas.ais(k=1)
                        te = meas.te(k=1)
                        mc = meas.mean_correlation()
                        cv = meas.current_variance()

                        for task in TASKS:
                            if task == "parity":
                                X, y = parity_task(n_steps=5000, history_length=2, input_dim=n_inputs, seed=trial)
                            elif task == "counting_task":
                                X, y = count_ones_in_window_task(n_steps=5000, history_length=5, input_dim=n_inputs, seed=trial)
                            else:
                                X, y = prep_task(task, X_full, history_length=10,act=True)
                           
                            n_inputs = X.shape[1]
                            
                            n_train = int(len(y) * 0.8)
                            X_train, y_train = X[:n_train], y[:n_train]
                            X_test, y_test = X[n_train:], y[n_train:]

                            trainer_train = build_reservoir(
                                input_dim=n_inputs,
                                reservoir_size=100,
                                spectral_radius=sr,
                                connectivity=0.1,
                                leak_rate=1.0,
                                scale=scale,
                                seed=trial,
                                sparse_mapping=False,
                            )

                            trainer_train.train(X_train.copy(), y_train.copy())
                            y_pred = trainer_train.predict(X_test.copy()) 
                            
                            nmse_error = nmse(y_test, y_pred)
                            results.append(
                                        dict(

                                            process_count = X.shape[1],
                                            task = task,
                                            trial = trial,
                                            scale = scale,
                                            spectral_radius=sr,
                                            average_state_entropy= ase,
                                            variance_activation_derivatives = vad,
                                            participation_ratio = pr,
                                            active_information_storage = ais,
                                            transfer_entropy = te,
                                            mean_correlation = mc,
                                            nmse_error = nmse_error,

                                            
                                        )
                                    )
                            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv("../../data/ALL_MEASURES_NMSE_CORR_DATA.csv", index=False)


if __name__ == "__main__":
    main()
