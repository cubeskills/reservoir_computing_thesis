import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from itertools import combinations

ALL_POSSIBLE_MEASURES = [
    "average_state_entropy", "variance_activation_derivatives",
    "participation_ratio", "active_information_storage",
    "mean_correlation", "transfer_entropy",
]
TARGET_COL = "nmse_error"
SPECTRAL_RANGES = {
    "Sub-critical": (0.2, 0.7),
    "Critical": (0.7, 1.3),
    "Super-critical": (1.3, 1.8),
}
plt.rcParams.update({
    'axes.labelsize': 20,   # x and y labels
    'axes.titlesize': 20,   # plot titles
    'xtick.labelsize': 16,  # x tick labels
    'ytick.labelsize': 16,  # y tick labels
    'legend.fontsize': 16,  # legends
    'figure.titlesize': 22  # suptitle
})


custom_cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def generate_regression_plots(df_full: pd.DataFrame, available_measures: list, group_by_col: str, output_dir: str, r2_scores_list: list):
    settings = {
        'process_count': {'title_vs': "Input Count", 'filename_part': "by_input_count"},
        'scale': {'title_vs': "Scale", 'filename_part': "by_scale"},
        'spectral_radius': {'title_vs': "Spectral Radius", 'filename_part': "by_sr"}
    }
    s = settings[group_by_col]

    # Define groups/conditions
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
            continue

        for i, task_name in enumerate(all_tasks):
            task_df = subset_df[subset_df['task'] == task_name]
            if len(task_df) < 5:
                continue

            try:
                X, y = task_df[available_measures], task_df[TARGET_COL]
                if len(X) == 0 or X.isnull().all().all():
                    continue
                    
                X_scaled = StandardScaler().fit_transform(X)
                model = LinearRegression()
                cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), scoring='r2')

                r2 = np.mean(cv_scores)
                r2_matrix[i, j] = r2
                
                r2_scores_list.append({
                    "condition_type": s['title_vs'],
                    "condition_value": condition_value,
                    "task": task_name,
                    "r2_score": r2
                })
            except:
                continue

    if np.isnan(r2_matrix).all():
        return

    heatmap_df = pd.DataFrame(
        r2_matrix,
        index=[task.replace('_', ' ').title() for task in all_tasks],
        columns=condition_names
    ).dropna(axis=0, how='all').dropna(axis=1, how='all')

    if heatmap_df.empty:
        return

    plt.figure(figsize=(max(8, len(heatmap_df.columns) * 1.5), max(6, len(heatmap_df.index) * 0.8)))
    #valid_data = heatmap_df.values[~np.isnan(heatmap_df.values)]
    #vmin, vmax = (np.min(valid_data), np.max(valid_data)) if len(valid_data) > 0 else (0, 1)
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

    plt.title(f"Linear Regression Performance (R² Scores)\nGrouped by {s['title_vs']}")
    plt.xlabel(s['title_vs'])
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"6_linreg_r2_heatmap_{s['filename_part']}.pdf"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_global_regression_plots(df_full: pd.DataFrame, available_measures: list, output_dir: str):
    all_results = []

    for task_name in df_full['task'].unique():
        task_df = df_full[df_full['task'] == task_name].copy()
        if len(task_df) < 5:
            continue

        try:
            X, y = task_df[available_measures], task_df[TARGET_COL]
            if len(X) == 0 or X.isnull().all().all():
                continue
                
            X_scaled = StandardScaler().fit_transform(X)
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), scoring='r2')

            r2_mean = np.mean(cv_scores)
            r2_std = np.std(cv_scores)
            
            all_results.append({
                "Task": task_name.replace('_',' ').title(),
                "R²": r2_mean,
                "R²_std": r2_std
            })
        except:
            continue

    if not all_results:
        return

    results_df = pd.DataFrame(all_results).set_index("Task")
    heatmap_df = results_df[["R²"]]
    #valid_data = heatmap_df.values[~np.isnan(heatmap_df.values)]
    #vmin, vmax = (np.min(valid_data), np.max(valid_data)) if len(valid_data) > 0 else (0, 1)
    vmin, vmax = 0, 1
    plt.figure(figsize=(8, max(6, len(heatmap_df) * 0.8)))
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
    
    plt.title(f"Global Linear Regression Performance (R² Scores)")
    plt.xlabel("Overall Performance")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_linreg_global_heatmap.pdf"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_summary_figures(r2_scores_df: pd.DataFrame, output_dir: str):
    if r2_scores_df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(r2_scores_df['r2_score'], bins=50, kde=True, color='lightcoral', stat='density')
    plt.title(f'Distribution of R² Scores - Linear Regression', fontsize=24)
    plt.xlabel('R² Score', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_r2_score_distribution_linreg.pdf"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

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

    df_full = load_and_prepare_data(DATA_FILE_PATH)
    available_measures = [m for m in ALL_POSSIBLE_MEASURES if m in df_full.columns]
    
    r2_scores_list = []
    
    for group in ['process_count', 'scale', 'spectral_radius']:
        generate_regression_plots(df_full, available_measures, group, FIGURE_DIR, r2_scores_list)

    generate_global_regression_plots(df_full, available_measures, FIGURE_DIR)
    
    if r2_scores_list:
        plot_summary_figures(pd.DataFrame(r2_scores_list), FIGURE_DIR)
        r2_df = pd.DataFrame(r2_scores_list)
        summary_df = summarize_r2_by_condition(r2_df)
        print(summary_df)

if __name__ == "__main__":
    main()
