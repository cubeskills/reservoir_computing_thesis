import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    "Sub-critical (sr < 0.7)": (0.2, 0.7),
    "Edge of Chaos (0.7 <= sr <= 1.3)": (0.7, 1.3),
    "Super-critical (sr > 1.3)": (1.3, 1.8),
}

def load_and_prepare_data(file_path: str, input_type: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.rename(columns={'te': 'transfer_entropy'}, inplace=True)
    df['input_type'] = input_type
    return df

def analyze_correlations(df: pd.DataFrame, available_measures: list, input_type: str):
    valid_cols = available_measures + [TARGET_COL]
    df_filtered = df[valid_cols].dropna()
    corr_matrix = df_filtered.corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f"Global Spearman Correlation ({input_type.upper()} Input)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_conditional_correlations_sr(df: pd.DataFrame, available_measures: list, input_type: str, task_name: str):
    valid_cols = available_measures + [TARGET_COL]
    results = {}
    for name, (low, high) in SPECTRAL_RANGES.items():
        range_df = df[(df['spectral_radius'] >= low) & (df['spectral_radius'] < high)]
        corr_matrix = range_df[valid_cols].dropna().corr(method='spearman')
        results[name] = corr_matrix.get(TARGET_COL, pd.Series(np.nan)).drop(TARGET_COL, errors='ignore')
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    sns.heatmap(results_df, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f"NMSE Spearman Correlation vs SR ({input_type.upper()}, Task: {task_name.title()})", fontsize=16)
    plt.ylabel("Reservoir Measures", fontsize=12)
    plt.xlabel("Spectral Radius Regime", fontsize=12)
    plt.tight_layout()
    plt.show()

def analyze_correlations_by_input_count(df: pd.DataFrame, available_measures: list, input_type: str, task_name: str):
    valid_cols = available_measures + [TARGET_COL]
    input_counts = sorted(df['process_count'].unique())
    results = {}
    for count in input_counts:
        count_df = df[df['process_count'] == count]
        corr_matrix = count_df[valid_cols].dropna().corr(method='spearman')
        col_name = f"Inputs: {count}"
        results[col_name] = corr_matrix.get(TARGET_COL, pd.Series(np.nan)).drop(TARGET_COL, errors='ignore')
    results_df = pd.DataFrame(results).reindex(available_measures)
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    sns.heatmap(results_df, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f"NMSE Spearman Correlation vs Input Count ({input_type.upper()}, Task: {task_name.title()})", fontsize=16)
    plt.ylabel("Reservoir Measures", fontsize=12)
    plt.xlabel("Number of Input Processes", fontsize=12)
    plt.tight_layout()
    plt.show()

def analyze_correlations_by_scale(df: pd.DataFrame, available_measures: list, input_type: str, task_name: str):
    valid_cols = available_measures + [TARGET_COL]
    scales = sorted(df['scale'].unique())
    results = {}
    for scale in scales:
        scale_df = df[df['scale'] == scale]
        corr_matrix = scale_df[valid_cols].dropna().corr(method='spearman')
        col_name = f"Scale: {scale}"
        results[col_name] = corr_matrix.get(TARGET_COL, pd.Series(np.nan)).drop(TARGET_COL, errors='ignore')
    results_df = pd.DataFrame(results).reindex(available_measures)
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    sns.heatmap(results_df, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f"NMSE Spearman Correlation vs Input Scale ({input_type.upper()}, Task: {task_name.title()})", fontsize=16)
    plt.ylabel("Reservoir Measures", fontsize=12)
    plt.xlabel("Input Scale", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    home_dir = "../../data"
    storing = "../../figures"
    data_files = {
        "rp": "ALL_MEASURES_NMSE_CORR_DATA.csv",
    }
    for input_type, file_name in data_files.items():
        os.makedirs(os.path.join(storing, input_type), exist_ok=True)
        file_path = os.path.join(home_dir, file_name)
        df_full = load_and_prepare_data(file_path, input_type)
        available_measures = [m for m in ALL_POSSIBLE_MEASURES if m in df_full.columns]
        for task in df_full['task'].unique():
            task_df = df_full[df_full['task'] == task].copy()
            analyze_conditional_correlations_sr(
                df=task_df,
                available_measures=available_measures,
                input_type=input_type,
                task_name=task
            )
            analyze_correlations_by_input_count(
                df=task_df,
                available_measures=available_measures,
                input_type=input_type,
                task_name=task
            )
            analyze_correlations_by_scale(
                df=task_df,
                available_measures=available_measures,
                input_type=input_type,
                task_name=task
            )
