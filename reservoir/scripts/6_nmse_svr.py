import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression  # (optional baseline)

ALL_POSSIBLE_MEASURES = [
    "average_state_entropy",
    "variance_activation_derivatives",
    "participation_ratio",
    "active_information_storage",
    "mean_correlation",
    "transfer_entropy",
]
TARGET_COL = "nmse_error"
SPECTRAL_RANGES = {
    "Sub-critical": (0.2, 0.7),
    "Critical": (0.7, 1.3),
    "Super-critical": (1.3, 1.8),
}

SVR_C = 50.0
SVR_EPSILON = 0.02   
SVR_GAMMA = "scale"
CV_MAX_SPLITS = 5
RANDOM_STATE = 42
Y_STD_FLOOR = 1e-8  

plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})
custom_cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def build_svr_model():
    """
    SVR with per-fold X scaling and per-fold y scaling.
    epsilon is small in the standardized-y space so tiny targets are not flattened.
    """
    svr = SVR(kernel='rbf', C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA)
    pipe = make_pipeline(
        StandardScaler(),   
        svr
    )
    model = TransformedTargetRegressor(
        regressor=pipe,
        transformer=StandardScaler() 
    )
    return model


def generate_regression_plots(df_full: pd.DataFrame, available_measures: list,
                              group_by_col: str, output_dir: str, r2_scores_list: list):
    settings = {
        'process_count': {'title_vs': "Input Count", 'filename_part': "by_input_count"},
        'scale': {'title_vs': "Scale", 'filename_part': "by_scale"},
        'spectral_radius': {'title_vs': "Spectral Radius", 'filename_part': "by_sr"}
    }
    s = settings[group_by_col]

    print(f"\nProcessing {s['title_vs']}...")

    if group_by_col == 'spectral_radius':
        groups = list(SPECTRAL_RANGES.items())
        condition_names = [name for name, _ in groups]
    else:
        unique_values = sorted(df_full[group_by_col].unique())
        groups = [(str(val), val) for val in unique_values]
        condition_names = [str(val) for val in unique_values]

    all_tasks = sorted(df_full['task'].unique())
    r2_matrix = np.full((len(all_tasks), len(groups)), np.nan)

    for j, group in enumerate(groups):
        if group_by_col == 'spectral_radius':
            name, (low, high) = group
            subset_df = df_full[(df_full['spectral_radius'] >= low) & (df_full['spectral_radius'] < high)]
            condition_value = name
        else:
            name, actual_value = group
            subset_df = df_full[df_full[group_by_col] == actual_value]
            condition_value = name

        if len(subset_df) < 10:
            print(f"    Skipping {condition_value} — insufficient data")
            continue

        for i, task_name in enumerate(all_tasks):
            task_df = subset_df[subset_df['task'] == task_name]
            if len(task_df) < 5:
                continue

            try:
                X = task_df[available_measures].values
                y = task_df[TARGET_COL].values
                if np.std(y) < Y_STD_FLOOR:
                    continue

                n_splits = min(CV_MAX_SPLITS, len(X))
                if n_splits < 2:
                    continue
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

                model = build_svr_model()
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                r2 = float(np.mean(cv_scores))
                r2_matrix[i, j] = r2

                r2_scores_list.append({
                    "condition_type": s['title_vs'],
                    "condition_value": condition_value,
                    "task": task_name,
                    "r2_score": r2
                })

            except Exception as e:
                print(f"      {task_name} @ {condition_value}: skipped due to error: {e}")
                continue

    if np.isnan(r2_matrix).all():
        print(f"No valid data for {s['title_vs']}, skipping heatmap")
        return

    heatmap_df = pd.DataFrame(
        r2_matrix,
        index=[task.replace('_', ' ').title() for task in all_tasks],
        columns=condition_names
    ).dropna(axis=0, how='all').dropna(axis=1, how='all')

    if heatmap_df.empty:
        print(f"No valid data remaining for {s['title_vs']} after cleaning")
        return

    plt.figure(figsize=(max(8, len(heatmap_df.columns) * 1.5), max(6, len(heatmap_df.index) * 0.8)))
    vmin, vmax = 0, 1
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.3f',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={'label': 'R² Score'},
        mask=pd.isna(heatmap_df)
    )

    plt.title(f"RBF SVR Performance (R² Scores)\nGrouped by {s['title_vs']}")
    plt.xlabel(s['title_vs'])
    plt.ylabel("Tasks")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"6_svr_r2_heatmap_{s['filename_part']}.pdf")
    out2 = os.path.join(output_dir, f"6_svr_r2_heatmap_{s['filename_part']}.png")
    # png save
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved heatmap: {os.path.basename(out_path)}")


def generate_global_regression_plots(df_full: pd.DataFrame, available_measures: list, output_dir: str):
    print("\nGenerating global regression plots...")
    all_results = []

    for task_name in sorted(df_full['task'].unique()):
        task_df = df_full[df_full['task'] == task_name].copy()
        print(f"  Task {task_name}: {len(task_df)} samples")
        if len(task_df) < 5:
            print(f"    Skipping - insufficient data")
            continue

        try:
            X = task_df[available_measures].values
            y = task_df[TARGET_COL].values

            if np.std(y) < Y_STD_FLOOR:
                print("    Skipping - target has ~zero variance (R² not informative)")
                continue

            n_splits = min(CV_MAX_SPLITS, len(X))
            if n_splits < 2:
                continue
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

            model = build_svr_model()
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

            r2_mean = float(np.mean(cv_scores))
            r2_std = float(np.std(cv_scores))

            all_results.append({
                "Task": task_name.replace('_', ' ').title(),
                "R²": r2_mean,
                "R²_std": r2_std
            })
            print(f"    R² = {r2_mean:.3f} ± {r2_std:.3f}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    if not all_results:
        print("No valid results for global plots")
        return

    results_df = pd.DataFrame(all_results).set_index("Task")
    heatmap_df = results_df[["R²"]]

    plt.figure(figsize=(8, max(6, len(heatmap_df) * 0.8)))
    vmin, vmax = 0, 1
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.3f',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={'label': 'R² Score'},
        mask=pd.isna(heatmap_df)
    )

    plt.title("Global RBF SVR Performance (R² Scores)")
    plt.xlabel("Overall Performance")
    plt.ylabel("Tasks")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "6_svr_global_heatmap.pdf")
    out2 = os.path.join(output_dir, "6_svr_global_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved global heatmap: {os.path.basename(out_path)}")


def plot_summary_figures(r2_scores_df: pd.DataFrame, output_dir: str):
    if r2_scores_df.empty:
        print("No R² scores to plot")
        return

    print(f"Plotting summary figures for {len(r2_scores_df)} R² scores...")
    plt.figure(figsize=(10, 6))
    sns.histplot(r2_scores_df['r2_score'], bins=50, kde=True, color='skyblue', stat='density')
    plt.title('Distribution of R² Scores', fontsize=24)
    plt.xlabel('R² Score', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "6_r2_score_distribution_svr.pdf")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()
    print(f"Saved R² distribution plot: {os.path.basename(out_path)}")


def summarize_r2_by_condition(r2_scores_df: pd.DataFrame):
    if r2_scores_df.empty:
        return None
    summary = (
        r2_scores_df
        .groupby("condition_type")["r2_score"]
        .agg(["mean", "var", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return summary


def main():
    DATA_DIR = "../../data"
    FIGURE_DIR = "../../figures"
    DATA_FILE_PATH = os.path.join(DATA_DIR, "ALL_MEASURES_NMSE_CORR_DATA.csv")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("Loading data...")
    df_full = load_and_prepare_data(DATA_FILE_PATH)
    available_measures = [m for m in ALL_POSSIBLE_MEASURES if m in df_full.columns]

    print(f"Data shape: {df_full.shape}")
    print(f"Available measures: {available_measures}")
    print(f"Unique tasks: {sorted(df_full['task'].unique())}")
    print(f"Unique scales: {sorted(df_full['scale'].unique())}")
    print(f"Unique process_counts: {sorted(df_full['process_count'].unique())}")
    print(f"Spectral radius range: {df_full['spectral_radius'].min():.2f} - {df_full['spectral_radius'].max():.2f}")

    r2_scores_list = []

    for group in ['process_count', 'scale', 'spectral_radius']:
        generate_regression_plots(df_full, available_measures, group, FIGURE_DIR, r2_scores_list)

    print("\nGenerating global plots...")
    generate_global_regression_plots(df_full, available_measures, FIGURE_DIR)

    if r2_scores_list:
        r2_df = pd.DataFrame(r2_scores_list)
        plot_summary_figures(r2_df, FIGURE_DIR)
        summary_df = summarize_r2_by_condition(r2_df)
        print("\n=== Summary of R² by Condition Type ===")
        print(summary_df)
    else:
        print("\nNo valid R² scores generated!")

    print(f"\nAnalysis complete. Figures saved to '{FIGURE_DIR}' directory.")


if __name__ == "__main__":
    main()
