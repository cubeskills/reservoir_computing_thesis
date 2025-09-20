import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from utils.simulation_tools import compute_random_processes, build_reservoir, prep_task
from utils.measures import Measures
from utils.metrics import nmse
from utils.sequence_generator import parity_task, count_ones_in_window_task

seed = 42
N_STEPS = 5000
process_dims = [1, 3, 5, 7]
processes = [compute_random_processes(dim, n_total_steps=N_STEPS, lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=seed) for dim in process_dims]
X_rp_sets = [p.T for p in processes]

spectral_radii = np.linspace(0.2, 1.8, 15)
scales = np.logspace(-3, 1, 15)

RP_TASKS = ["delay", "delay_polynomial", "polynomial", "narma"]
GEN_TASKS = ["parity", "count_ones_in_window"]
trials = 10
n_jobs = -1

def evaluate_performance(X_task, y_task, sr, scale, trial):
    n_train = int(len(y_task) * 0.8)
    X_train, y_train = X_task[:n_train], y_task[:n_train]
    X_test, y_test = X_task[n_train:], y_task[n_train:]
    trainer = build_reservoir(input_dim=X_task.shape[1], reservoir_size=100, spectral_radius=sr,
                              connectivity=0.1, leak_rate=1.0, scale=scale, seed=trial)
    trainer.train(X_train.copy(), y_train.copy())
    y_pred = trainer.predict(X_test.copy())
    return nmse(y_test, y_pred) if np.var(y_test) >= 1e-9 else np.mean((y_test - y_pred)**2)

def process_task(X_task, y_task, sr, scale, trial, shared_measures, task_name, process_count):
    base_nmse = evaluate_performance(X_task, y_task, sr, scale, trial)
    return {
        **shared_measures,
        'task': task_name,
        'process_count': process_count,
        'trial': trial,
        'spectral_radius': sr,
        'scale': scale,
        'base_nmse': base_nmse
    }

def compute_shared_measures(X, trial, sr, scale):
    meas_reservoir = build_reservoir(input_dim=X.shape[1], reservoir_size=100, spectral_radius=sr,
                                    connectivity=0.1, leak_rate=1.0, scale=scale, seed=trial)
    states = meas_reservoir.collect_states(X)
    meas = Measures(meas_reservoir.current_history, states)
    return {
        'participation_ratio': meas.participation_ratio(),
        'variance_activation_derivatives': meas.variance_of_activation_derivatives(),
        'average_state_entropy': meas.average_state_entropy(),
        'active_information_storage': meas.ais(k=1),
        'transfer_entropy': meas.te(k=1),
        'mean_correlation': meas.mean_correlation(),
    }

def main():
    results = []

    def task_worker(trial, x_set, scale, sr, task_name, X_full, y_full, process_count):
        shared_measures = compute_shared_measures(x_set, trial, sr, scale)
        return process_task(X_full, y_full, sr, scale, trial, shared_measures, task_name, process_count)

    jobs = []

    for trial in range(trials):
        for x_set in X_rp_sets:
            process_count = x_set.shape[1]
            for scale in scales:
                for sr in spectral_radii:
                    for task in RP_TASKS:
                        X_full, y_full = prep_task(task, x_set, history_length=10, act=True)
                        jobs.append(delayed(task_worker)(trial, x_set, scale, sr, task, X_full, y_full, process_count))

        for dim in process_dims:
            X_parity, y_parity = parity_task(n_steps=N_STEPS, history_length=3, input_dim=dim, seed=trial)
            X_count, y_count = count_ones_in_window_task(n_steps=N_STEPS, history_length=5, input_dim=dim, seed=trial)
            task_data_map = {"parity": (X_parity, y_parity), "count_ones_in_window": (X_count, y_count)}
            for scale in scales:
                for sr in spectral_radii:
                    for task in GEN_TASKS:
                        X_full, y_full = task_data_map[task]
                        jobs.append(delayed(task_worker)(trial, X_parity, scale, sr, task, X_full, y_full, dim))

    results = Parallel(n_jobs=n_jobs)(tqdm(jobs, desc="Parallel Sweep"))
    pd.DataFrame(results).to_csv("../../data/RASTER_DATA_GEN_GLOBAL.csv", index=False)

if __name__ == "__main__":
    main()
