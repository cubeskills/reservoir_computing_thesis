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
from sklearn.inspection import permutation_importance


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from utils.simulation_tools import build_reservoir, compute_random_processes, prep_task
from utils.metrics import nmse
from utils.measures import Measures
from utils.sequence_generator import parity_task, count_ones_in_window_task

SR_INITS = [0.2, 1.8]
SCALE_INITS = [1e-3, 1e1]
EPOCHS = 25
SEED = 42
CSV_SWEEP_FILE = "../../data/RASTER_DATA_GEN.csv"

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

def format_colorbar(cbar, label):
    """Ensure colorbar uses the same readable typography as the rest."""
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

def load_data(file):
    if not os.path.exists(file):
        print(f"Warning: Data file '{file}' not found. Cannot proceed.")
        return pd.DataFrame()
    return pd.read_csv(file)

def plot_svm_permutation_importance(
    svm_sr,
    svm_scale,
    X_scaled,
    y_sr,
    y_scale,
    feature_names,
    n_repeats=20,
    scoring="accuracy",   
    random_state=SEED,
    outdir="../../figures",
    basename="svm_permutation_importance"
):
    
    os.makedirs(outdir, exist_ok=True)

    def _perm_df(estimator, X, y):
        r = permutation_importance(
            estimator,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )
        df_imp = pd.DataFrame(
            {
                "feature": feature_names,
                "mean": r.importances_mean,
                "std": r.importances_std,
            }
        ).sort_values("mean", ascending=True, ignore_index=True)
        return df_imp

    df_sr = _perm_df(svm_sr, X_scaled, y_sr)
    df_scale = _perm_df(svm_scale, X_scaled, y_scale)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    for ax, title, df in [
        (axes[0], "SVM — SR direction", df_sr),
        (axes[1], "SVM — Scale direction", df_scale),
    ]:
        bars = ax.barh(df["feature"], df["mean"], xerr=df["std"], capsize=4)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Mean decrease in score (permutation)")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()  

        for rect, mu, sd in zip(bars, df["mean"], df["std"]):
            x = rect.get_width()
            y = rect.get_y() + rect.get_height() / 2
            ha = "left" if x >= 0 else "right"
            offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.text(x + (offset if x >= 0 else -offset), y, f"{mu:.3f}±{sd:.3f}",
                    va="center", ha=ha)

    plt.tight_layout()
    png_path = os.path.join(outdir, f"{basename}.png")
    pdf_path = os.path.join(outdir, f"{basename}.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.show()

    return df_sr, df_scale


def create_targets_percentile_only(df, percentile=0.03):
    df_processed = df.copy()
    sr_directions = pd.Series(index=df.index, dtype=object)
    scale_directions = pd.Series(index=df.index, dtype=object)

    for (task, p_count), group in df_processed.groupby(['task', 'process_count']):
        if group.empty:
            continue
        optimum_row = group.loc[group['base_nmse'].idxmin()]
        sr_opt = optimum_row['spectral_radius']
        scale_opt = optimum_row['scale']

        performance_threshold = group['base_nmse'].quantile(percentile)
        stay_zone_indices = group[group['base_nmse'] <= performance_threshold].index
        move_zone_indices = group.index.difference(stay_zone_indices)

        sr_directions.loc[stay_zone_indices] = 'stay'
        scale_directions.loc[stay_zone_indices] = 'stay'

        if not move_zone_indices.empty:
            move_group = group.loc[move_zone_indices]
            sr_directions.loc[move_zone_indices] = np.select(
                [sr_opt > move_group['spectral_radius'], sr_opt < move_group['spectral_radius']],
                ['increase', 'decrease'],
                default='increase'
            )
            scale_directions.loc[move_zone_indices] = np.select(
                [scale_opt > move_group['scale'], scale_opt < move_group['scale']],
                ['increase', 'decrease'],
                default='increase'
            )

    df_processed['sr_direction_global'] = sr_directions
    df_processed['scale_direction_global'] = scale_directions
    return df_processed


def evaluate_performance(task_name, process_data, sr, scale, seed):
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

def plot_error_landscape_with_trajectories(csv_file, task_name, process_count, trajectories, EPOCHS):
    df = pd.read_csv(csv_file)
    task_df = df[(df["task"] == task_name) & (df["process_count"] == process_count)]
    if task_df.empty:
        print(f"No data for task={task_name}, process_count={process_count} in {csv_file}")
        return

    performance_grid = (
        task_df.groupby(["scale", "spectral_radius"])["base_nmse"]
               .mean()
               .reset_index()
    )
    pivoted_df = (
        performance_grid.pivot(index="spectral_radius", columns="scale", values="base_nmse")
                        .sort_index(axis=0)
                        .sort_index(axis=1)
    )

    fig, ax = plt.subplots()
    hm = sns.heatmap(
        pivoted_df,
        ax=ax,
        cmap="viridis_r",
        annot=False,
        vmin=0, vmax=1,         
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Mean NMSE"}
    )

    min_nmse_val = pivoted_df.min().min()
    min_pos = pivoted_df.stack().idxmin()
    min_sr, min_scale = min_pos
    ax.scatter(
        pivoted_df.columns.get_loc(min_scale) + 0.5,
        pivoted_df.index.get_loc(min_sr) + 0.5,
        marker="*",
        color="red",
        s=200,
        zorder=5,
        label=f"Best NMSE: {min_nmse_val:.3f}",
    )

    line_styles = ["-", "--", "-.", ":"]
    for idx, (label, (sr_hist, sc_hist)) in enumerate(trajectories.items()):
        style = line_styles[idx % len(line_styles)]
        sc_indices = [pivoted_df.columns.get_indexer([sc], method="nearest")[0] + 0.5 for sc in sc_hist[:EPOCHS]]
        sr_indices = [pivoted_df.index.get_indexer([sr], method="nearest")[0] + 0.5 for sr in sr_hist[:EPOCHS]]
        ax.plot(sc_indices, sr_indices, style, marker="o", markersize=5, label=label)

    ax.set_title(f'Average NMSE across trials & input counts\nTask: "{task_name}" (Processes: {process_count})')
    ax.set_xlabel("Scale (log-grid)")
    ax.set_ylabel("Spectral Radius")

    ax.set_xticklabels([f"{val:.3f}" for val in pivoted_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{val:.2f}" for val in pivoted_df.index], rotation=0)
    cbar = hm.collections[0].colorbar
    format_colorbar(cbar, "Mean NMSE")
    leg = ax.legend()
    for txt in leg.get_texts():
        txt.set_color("black")

    plt.tight_layout()
    plt.savefig("../../figures/8_nmse_convergence_delay_heatmap_memory_tasks.pdf")
    plt.show()
    plt.close()



def plot_task_nmse_heatmaps(csv_file: str, outdir: str = "../../figures"):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_file)

    required_cols = {"task", "spectral_radius", "scale", "base_nmse"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Missing columns in '{csv_file}': {missing}")
        return

    agg = (
        df.groupby(["task", "spectral_radius", "scale"], as_index=False)["base_nmse"]
          .mean()
          .rename(columns={"base_nmse": "mean_nmse"})
    )

    tasks = sorted(agg["task"].unique())
    for task in tasks:
        sub = agg[agg["task"] == task].copy()
        pivot = (
            sub.pivot(index="spectral_radius", columns="scale", values="mean_nmse")
               .sort_index(axis=0)
               .sort_index(axis=1)
        )

        if pivot.empty:
            print(f"[INFO] No data to plot for task={task}")
            continue

        fig, ax = plt.subplots() 
        hm = sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis_r",
            annot=False,
            vmin=0, vmax=1,               
            cbar_kws={"label": "Mean NMSE"},
            linewidths=0.4,
            linecolor="white"
        )

        # mark best NMSE
        min_pos = pivot.stack().idxmin()
        min_sr, min_scale = min_pos
        ax.scatter(
            pivot.columns.get_loc(min_scale) + 0.5,
            pivot.index.get_loc(min_sr) + 0.5,
            marker="*",
            color="red",
            s=200,
            zorder=5,
            label=f"Best NMSE: {pivot.min().min():.3f}"
        )

        ax.set_title(f'Average NMSE across trials & input counts\nTask: "{task}"')
        ax.set_xlabel("Scale (log-grid)")
        ax.set_ylabel("Spectral Radius")
        ax.set_xticklabels([f"{val:.3f}" for val in pivot.columns], rotation=45, ha="right")
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index], rotation=0)

        cbar = hm.collections[0].colorbar
        format_colorbar(cbar, "Mean NMSE")

        leg = ax.legend()
        for txt in leg.get_texts():
            txt.set_color("black")

        plt.tight_layout()
        out_path = os.path.join(outdir, f"NMSE_grid_{task}.pdf")
        plt.savefig(out_path)
        plt.show()
        plt.close()



def main():
    file = CSV_SWEEP_FILE
    df = load_data(file)
    if df.empty:
        return

    df_processed = df[df['task'].isin(['delay',"count_ones_in_window"])]
    df_processed = df_processed[df_processed['process_count'].isin([1, 5, 7])]

    df_processed = create_targets_percentile_only(df_processed, percentile=0.03)
    print(f"Data loaded and processed. Number of samples: {len(df_processed)}")

    MEASURES = [
        "average_state_entropy",
        "variance_activation_derivatives",
        "participation_ratio",
        "active_information_storage",
        "transfer_entropy",
        "mean_correlation",
    ]
    
    X_train_svm = df_processed[MEASURES].values
    scaler = StandardScaler().fit(X_train_svm)
    X_train_svm_scaled = scaler.transform(X_train_svm)

    y_sr = df_processed["sr_direction_global"].values
    svm_sr = SVC(kernel="rbf", C=100, gamma="scale").fit(X_train_svm_scaled, y_sr)
    print("SVM for Spectral Radius trained.")

    y_scale = df_processed["scale_direction_global"].values
    svm_scale = SVC(kernel="rbf", C=100, gamma="scale").fit(X_train_svm_scaled, y_scale)
    print("SVM for Scale trained.")
    plot_svm_permutation_importance(
        svm_sr, svm_scale,
        X_train_svm_scaled, y_sr, y_scale,
        feature_names=MEASURES,
        n_repeats=20,         
        scoring="f1_macro",     
        outdir="../../figures",
        basename="svm_perm_importance"
    )

    
    """
    _ = plot_svm_permutation_importance(
            svm_sr, svm_scale,
            X_train_svm_scaled, y_sr, y_scale,
            feature_names=MEASURES,
            n_repeats=20,          
            scoring="f1_macro",    
            outdir="../../figures",
            basename="svm_perm_importance"
        )

    """
    sr_step = 0.05
    scale_step_factor = 1.5

    all_nmse_histories = {}
    trajectories = {}

    for sr_init in SR_INITS:
        for scale_init in SCALE_INITS:
            nmse_histories = {"delay": [],
                              "narma": [], 
                              "count_ones_in_window": []}
            sr_history, scale_history = [], []
            current_sr, current_scale = sr_init, scale_init

            init_label = fr"$\rho$={sr_init},$s$={scale_init}"
            print(f"Starting optimization with {init_label}")

            for epoch in tqdm(range(EPOCHS), desc="Optimizing Hyperparameters"):
                sr_history.append(current_sr)
                scale_history.append(current_scale)

                process = compute_random_processes(
                    3, n_total_steps=1000,
                    lower=-np.sqrt(3/4), upper=np.sqrt(3/4), seed=epoch
                ).T

                nmse_histories["delay"].append(
                    evaluate_performance("delay", process, current_sr, current_scale, SEED)
                )
                nmse_histories["narma"].append(
                    evaluate_performance("narma", process, current_sr, current_scale, SEED)
                )
                nmse_histories["count_ones_in_window"].append(
                    evaluate_performance("count_ones_in_window", process, current_sr, current_scale, SEED)
                )   
        

                meas_reservoir = build_reservoir(
                    input_dim=3, reservoir_size=100,
                    spectral_radius=current_sr, connectivity=0.1,
                    leak_rate=1.0, scale=current_scale, seed=SEED
                )
                states = meas_reservoir.collect_states(process)
                meas = Measures(meas_reservoir.current_history, states)

                live_measures_dict = {
                    "average_state_entropy": meas.average_state_entropy(),
                    "variance_activation_derivatives": meas.variance_of_activation_derivatives(),
                    "participation_ratio": meas.participation_ratio(),
                    "active_information_storage": meas.ais(k=1),
                    "transfer_entropy": meas.te(k=1),
                    "mean_correlation": meas.mean_correlation(),
                }
                live_measures_array = np.array([live_measures_dict[m] for m in MEASURES]).reshape(1, -1)
                live_measures_scaled = scaler.transform(live_measures_array)
                sr_direction = svm_sr.predict(live_measures_scaled)[0]
                scale_direction = svm_scale.predict(live_measures_scaled)[0]

                if sr_direction == "increase":
                    current_sr += sr_step
                elif sr_direction == "decrease":
                    current_sr -= sr_step
                if scale_direction == "increase":
                    current_scale *= scale_step_factor
                elif scale_direction == "decrease":
                    current_scale /= scale_step_factor

                current_sr = np.clip(current_sr, 0.1, 2.0)
                current_scale = np.clip(current_scale, 1e-4, 20.0)

            for task, hist in nmse_histories.items():
                if task not in all_nmse_histories:
                    all_nmse_histories[task] = {}
                all_nmse_histories[task][init_label] = hist
            trajectories[init_label] = (sr_history, scale_history)

    epochs_range = range(EPOCHS)
    color_cycle = plt.cm.tab10.colors
    line_styles = ["-", "--", "-.", ":"]

    for t_idx, (task, inits) in enumerate(all_nmse_histories.items()):
        color = color_cycle[t_idx % len(color_cycle)]
        for l_idx, (init_label, nmse_hist) in enumerate(inits.items()):
            style = line_styles[l_idx % len(line_styles)]
            plt.plot(
                epochs_range, nmse_hist,
                linestyle=style, marker="o",
                color=color,
                label=f"{task} ({init_label})"
            )

    plt.title("NMSE Convergence Across Different Initializations")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("../../figures/8_nmse_convergence_memory_tasks.pdf")
    plt.show()

    plot_error_landscape_with_trajectories(
        csv_file=CSV_SWEEP_FILE,
        task_name="narma",
        process_count=3,   
        trajectories=trajectories,
        EPOCHS=EPOCHS
    )

    plot_task_nmse_heatmaps(CSV_SWEEP_FILE, outdir="../../figures")

if __name__ == "__main__":
    main()
