import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

THEME = {
    "figure.figsize": (16, 12),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 14,         
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",    
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.0,
    "axes.prop_cycle": cycler("color", plt.cm.tab10.colors),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
}
sns.set_theme(style="whitegrid", rc=THEME)

def load_perturbation_results(file_path="../../data/perturbation_experiment_results.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    df = pd.read_csv(file_path)
    return df


def plot_averaged_trajectories(df, output_dir="../../figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # compute statistics across trial seeds
    grouped = df.groupby(['init_label', 'task', 'epoch']).agg({
        'nmse': ['mean', 'std'],
        'sr': 'mean',
        'scale': 'mean'
    }).reset_index()
    
    # flatten column names
    grouped.columns = ['init_label', 'task', 'epoch', 'nmse_mean', 'nmse_std', 'sr_mean', 'scale_mean']
    
    unique_init_labels = grouped['init_label'].unique()
    
    fig, axes = plt.subplots(len(unique_init_labels), 2, 
                           figsize=(16, 4*len(unique_init_labels)))
    if len(unique_init_labels) == 1:
        axes = axes.reshape(1, 2)
    
    colors = plt.cm.tab10.colors
    task_colors = {task: colors[i] for i, task in enumerate(grouped['task'].unique())}
    
    for row, init_label in enumerate(unique_init_labels):
        init_data = grouped[grouped['init_label'] == init_label]
        
        ax1 = axes[row, 0]
        ax2 = axes[row, 1]
        
        for task in init_data['task'].unique():
            task_data = init_data[init_data['task'] == task]
            
            # plot mean with confidence interval
            ax1.plot(task_data['epoch'], task_data['nmse_mean'], 
                    color=task_colors[task], label=task, 
                    linewidth=2, alpha=0.9)
            ax1.fill_between(task_data['epoch'], 
                            task_data['nmse_mean'] - task_data['nmse_std'],
                            task_data['nmse_mean'] + task_data['nmse_std'],
                            color=task_colors[task], alpha=0.2)
        
        ax1.axvline(x=30, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.axvline(x=55, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax1.set_title(f'Averaged Trajectory: {init_label}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('NMSE')
        ax1.set_yscale('log')
        
        # only show legend in first panel, positioned in lower left
        if row == 0:
            ax1.legend(loc='lower left')
        
        ax1.grid(True, alpha=0.3)
        
        # plot averaged hyperparameters
        param_data = init_data.drop_duplicates(['epoch'])
        ax2_twin = ax2.twinx()
        
        ax2.plot(param_data['epoch'], param_data['sr_mean'], 
                color='blue', linewidth=3, label='Spectral Radius')
        ax2_twin.plot(param_data['epoch'], param_data['scale_mean'], 
                     color='green', linewidth=3, label='Input Scale')
        
        ax2.axvline(x=30, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.axvline(x=55, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax2.set_title(f'Averaged Parameters: {init_label}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Spectral Radius', color='blue')
        ax2_twin.set_ylabel('Input Scale', color='green')
        ax2_twin.set_yscale('log')
        
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "9_averaged_perturbation_trajectories.pdf"))
    plt.close()

def main():
    
    df = load_perturbation_results()
    plot_averaged_trajectories(df)
        

if __name__ == "__main__":
    main()
