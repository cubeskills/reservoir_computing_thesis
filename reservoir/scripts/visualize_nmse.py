import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def plot_nmse_heatmap(df, task_name, process_count, output_dir="../../figures"):
    print(f"Generating plot for task: '{task_name}' with process count: {process_count}...")
    task_df = df[(df['task'] == task_name) & (df['process_count'] == process_count)]
    if task_df.empty:
        print(f"Warning: No data found for task='{task_name}' and process_count={process_count}. Skipping plot.")
        return
    performance_grid = task_df.groupby(['scale', 'spectral_radius'])['base_nmse'].mean().reset_index()
    pivoted_df = performance_grid.pivot(index='scale', columns='spectral_radius', values='base_nmse')
    pivoted_df = pivoted_df.sort_index(axis=0).sort_index(axis=1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    norm = LogNorm(vmin=pivoted_df.min().min() + 1e-9, vmax=pivoted_df.max().max())
    min_nmse_val = pivoted_df.min().min()
    min_nmse_pos = pivoted_df.stack().idxmin()
    min_scale, min_sr = min_nmse_pos
    sns.heatmap(
        pivoted_df,
        ax=ax,
        cmap='viridis_r',
        annot=False,
        norm=norm,
        cbar_kws={'label': 'Mean NMSE (Log Scale)'}
    )
    x_loc = pivoted_df.index.get_loc(min_scale)
    y_loc = pivoted_df.columns.get_loc(min_sr)
    ax.scatter(x_loc + 0.5, y_loc + 0.5, marker='*', color='red', s=200, zorder=5, label=f'Best NMSE: {min_nmse_val:.3f}')
    ax.set_title(f'NMSE Landscape for Task: "{task_name}" (Input Processes: {process_count})', fontsize=16, weight='bold')
    ax.set_ylabel('Spectral Radius', fontsize=12)
    ax.set_xlabel('Scale (log)', fontsize=12)
    ax.set_yticklabels([f'{val:.2f}' for val in pivoted_df.columns], rotation=45)
    ax.set_xticklabels([f'{val:.3f}' for val in pivoted_df.index], rotation=0)
    ax.legend()
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{output_dir}/nmse_landscape_{task_name}_pc{process_count}.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to '{filename}'")
    plt.show()

def main():
    csv_file_path = "../../data/RASTER_DATA_GEN.csv"
    TASK_TO_PLOT = 'delay'
    PROCESS_COUNT_TO_PLOT = 3
    if not os.path.exists(csv_file_path):
        print(f"Error: Data file not found at '{csv_file_path}'")
        print("Please run the data generation script first or update the file path.")
        return
    print(f"Loading data from '{csv_file_path}'...")
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
    plot_nmse_heatmap(df, TASK_TO_PLOT, PROCESS_COUNT_TO_PLOT)

if __name__ == "__main__":
    main()
