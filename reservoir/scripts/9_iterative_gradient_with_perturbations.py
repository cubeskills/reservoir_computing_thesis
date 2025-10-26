import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import os
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from reservoir.utils.simulation_tools import build_reservoir, compute_random_processes, prep_task
from reservoir.utils.metrics import nmse
from reservoir.utils.measures import Measures
from reservoir.utils.sequence_generator import parity_task, count_ones_in_window_task

sys.path.insert(0, os.path.dirname(__file__))
from helpers_gradient_prediction import load_data, create_targets_percentile_only

SR_INITS = [0.2, 1.8]
SCALE_INITS = [1e-3, 1e1]
INITIAL_EPOCHS = 30
PERTURBATION_EPOCHS = 25
N_REPETITIONS = 10
SEED = 42
CSV_SWEEP_FILE = "../../data/RASTER_DATA_GEN_GLOBAL.csv"

THEME = {
    "figure.figsize": (14, 10),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 20,         
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "axes.labelweight": "bold",    
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.frameon": True,
    "lines.linewidth": 2.5,
    "lines.markersize": 6.0,
    "axes.prop_cycle": cycler("color", plt.cm.tab10.colors),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "image.cmap": "viridis_r",
}
sns.set_theme(style="whitegrid", rc=THEME)

def evaluate_performance(task_name, process_data, sr, scale, seed, input_variance_factor=1.0):
    if task_name == "parity":
        X_task, y_task = parity_task(
            n_steps=5000,
            history_length=2,
            input_dim=process_data.shape[1],
            seed=seed,
            task="multi_sum"
        )
    elif task_name == "count_ones_in_window":
        X_task, y_task = count_ones_in_window_task(
            n_steps=5000,
            history_length=5,
            input_dim=process_data.shape[1],
            seed=seed
        )
    else:
        X_task, y_task = prep_task(task_name, process_data, history_length=10, act=True)

    # apply input variance perturbation to simulate environment change
    X_task = X_task * np.sqrt(input_variance_factor)
    
    n_inputs = X_task.shape[1]
    n_train = int(len(y_task) * 0.8)
    X_train, y_train = X_task[:n_train], y_task[:n_train]
    X_test, y_test = X_task[n_train:], y_task[n_train:]

    trainer = build_reservoir(
        input_dim=n_inputs,
        reservoir_size=100,
        spectral_radius=sr,
        connectivity=0.1,
        leak_rate=1.0,
        scale=scale,
        seed=seed,
    )
    trainer.train(X_train.copy(), y_train.copy())
    y_pred = trainer.predict(X_test.copy())
    return nmse(y_test, y_pred)

def run_optimization_phase(sr_init, scale_init, epochs, svm_sr, svm_scale, scaler, measures, 
                          process_seed, sr_perturbation=0.0, input_variance_factor=1.0, 
                          phase_name="initial"):
    
    sr_step = 0.05
    scale_step_factor = 1.5
    
    nmse_histories = {
        "delay": [], "narma": [], "count_ones_in_window": [],
        "parity": [], "polynomial": [], "delay_polynomial": []
    }
    sr_history, scale_history = [], []
    current_sr, current_scale = sr_init, scale_init
    
    for epoch in tqdm(range(epochs), desc=f"{phase_name} phase"):
        sr_history.append(current_sr)
        scale_history.append(current_scale)
        
        process = compute_random_processes(
            3, n_total_steps=1000,
            lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=process_seed + epoch
        ).T

        # apply perturbation to simulate unknown reservoir changes with effective sr
        effective_sr = current_sr + sr_perturbation
        # sr need to be positive
        effective_sr = max(effective_sr, 0.05)  

        for task in nmse_histories.keys():
            nmse_val = evaluate_performance(
                task, process, effective_sr, current_scale, 
                SEED, input_variance_factor
            )
            nmse_histories[task].append(nmse_val)
        
        # compute reservoir measures to predict optimization directions with SVM
        meas_reservoir = build_reservoir(
            input_dim=3, reservoir_size=100,
            spectral_radius=effective_sr, connectivity=0.1,
            leak_rate=1.0, scale=current_scale, seed=SEED
        )
        
        process_scaled = process * np.sqrt(input_variance_factor)
        states = meas_reservoir.collect_states(process_scaled)
        meas = Measures(meas_reservoir.current_history, states)
        
        live_measures_dict = {
            "average_state_entropy": meas.average_state_entropy(),
            "variance_activation_derivatives": meas.variance_of_activation_derivatives(),
            "participation_ratio": meas.participation_ratio(),
            "active_information_storage": meas.ais(k=1),
            "transfer_entropy": meas.te(k=1),
            "mean_correlation": meas.mean_correlation(),
        }
        
        live_measures_array = np.array([live_measures_dict[m] for m in measures]).reshape(1, -1)
        live_measures_scaled = scaler.transform(live_measures_array)
        sr_direction = svm_sr.predict(live_measures_scaled)[0]
        scale_direction = svm_scale.predict(live_measures_scaled)[0]
        
        if sr_direction == "increase":
            current_sr += sr_step
        elif sr_direction == "decrease":
            current_sr -= sr_step
        
        # clip spectral radius to make it positive
        current_sr = np.clip(current_sr, 0.05, np.inf)
        
        if scale_direction == "increase":
            current_scale *= scale_step_factor
        elif scale_direction == "decrease":
            current_scale /= scale_step_factor
    
    return nmse_histories, sr_history, scale_history

def run_perturbation_experiment(trial_number):
    file = CSV_SWEEP_FILE
    df = load_data(file)
    if df.empty:
        return None

    df_processed = create_targets_percentile_only(df, percentile=0.05)
    measures = [
        "average_state_entropy", "variance_activation_derivatives", "participation_ratio",
        "active_information_storage", "transfer_entropy", "mean_correlation"
    ]
    
    # generate unique weight matrix seed from trial number
    weight_matrix_seed = SEED + trial_number * 100
    
    # train SVM models to predict optimization directions from reservoir measures
    X = df_processed[measures].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_sr = df_processed["sr_direction_global"].values
    y_scale = df_processed["scale_direction_global"].values
    
    svm_sr = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
    svm_scale = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
    svm_sr.fit(X_scaled, y_sr)
    svm_scale.fit(X_scaled, y_scale)
    
    results = []
    # for all spectral radius and scale initializations
    # 1. run initial optimization phase
    # 2. run perturbation phase 1 (increase sr, increase input variance)
    # 3. run perturbation phase 2 (decrease sr, decrease input variance)

    for sr_init in SR_INITS:
        for scale_init in SCALE_INITS:
            init_label = f"sr{sr_init}_scale{scale_init}"
            
            initial_nmse, initial_sr, initial_scale = run_optimization_phase(
                sr_init, scale_init, INITIAL_EPOCHS, svm_sr, svm_scale, 
                scaler, measures, weight_matrix_seed, 
                phase_name=f"Initial {init_label}"
            )
            
            final_sr = initial_sr[-1]
            final_scale = initial_scale[-1]
            
            pert1_nmse, pert1_sr, pert1_scale = run_optimization_phase(
                final_sr, final_scale, PERTURBATION_EPOCHS, svm_sr, svm_scale,
                scaler, measures, weight_matrix_seed + 1000, 
                sr_perturbation=0.5, input_variance_factor=10.0,
                phase_name=f"Pert1 {init_label}"
            )
            
            pert2_nmse, pert2_sr, pert2_scale = run_optimization_phase(
                pert1_sr[-1], pert1_scale[-1], PERTURBATION_EPOCHS, svm_sr, svm_scale,
                scaler, measures, weight_matrix_seed + 2000,
                sr_perturbation=-0.5, input_variance_factor=0.1,
                phase_name=f"Pert2 {init_label}"
            )
            
            for task in initial_nmse.keys():
                combined_nmse = initial_nmse[task] + pert1_nmse[task] + pert2_nmse[task]
                combined_sr = initial_sr + pert1_sr + pert2_sr
                combined_scale = initial_scale + pert1_scale + pert2_scale
                
                for epoch, (nmse_val, sr_val, scale_val) in enumerate(zip(combined_nmse, combined_sr, combined_scale)):
                    phase = "initial" if epoch < INITIAL_EPOCHS else ("pert1" if epoch < INITIAL_EPOCHS + PERTURBATION_EPOCHS else "pert2")
                    phase_epoch = epoch if phase == "initial" else (epoch - INITIAL_EPOCHS if phase == "pert1" else epoch - INITIAL_EPOCHS - PERTURBATION_EPOCHS)
                    
                    results.append({
                        "trial": trial_number,
                        "sr_init": sr_init,
                        "scale_init": scale_init,
                        "init_label": init_label,
                        "task": task,
                        "epoch": epoch,
                        "phase": phase,
                        "phase_epoch": phase_epoch,
                        "nmse": nmse_val,
                        "sr": sr_val,
                        "scale": scale_val
                    })
    
    return pd.DataFrame(results)

def main():
    all_results = []
    
    for rep in tqdm(range(N_REPETITIONS), desc="Weight matrix repetitions"):
        trial_number = rep + 1  
        print(f"Starting repetition {trial_number}/{N_REPETITIONS}")
        rep_results = run_perturbation_experiment(trial_number)
        if rep_results is not None:
            all_results.append(rep_results)
    
    if not all_results:
        return
    
    final_df = pd.concat(all_results, ignore_index=True)
    os.makedirs("../../data", exist_ok=True)
    final_df.to_csv("../../data/perturbation_experiment_results.csv", index=False)

if __name__ == "__main__":
    main()