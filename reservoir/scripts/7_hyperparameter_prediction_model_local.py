import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from helpers_gradient_prediction import (
    train_and_evaluate,
    plot_confusion_matrix,
    create_local_targets,
)
def load_data(file):
    if not os.path.exists(file):
        print(f"Warning: Data file '{file}' not found. Cannot proceed.")
        return pd.DataFrame()
    return pd.read_csv(file)

def plot_f1_matrix(y_true, y_pred, class_labels, title="F1 Score Matrix", storing=None, parameter=None, experiment=None):
    """
    Plot F1 scores in a matrix format similar to confusion matrix
    """
    f1_scores = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    
    f1_matrix = np.zeros((len(class_labels), len(class_labels)))
    np.fill_diagonal(f1_matrix, f1_scores)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                vmin=0, vmax=1)
    plt.title(title)
    plt.ylabel('True Class')
    plt.xlabel('F1 Score by Class')
    
    if storing:
        os.makedirs(storing, exist_ok=True)
        filename = f"{experiment}_experiment_svm_f1_matrix_{parameter}_local.png"
        #plt.savefig(os.path.join(storing, filename), bbox_inches='tight')
        print(f"Saved F1 matrix plot to {os.path.join(storing, filename)}")

    plt.show()
    plt.close()

def analyze_by_task(df_processed, measures, storing, experiment):
    """
    Analyze SVM performance for each task separately
    """
    tasks = df_processed['task'].unique()
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"ANALYZING TASK: {task}")
        print(f"{'='*50}")
        
        task_data = df_processed[df_processed['task'] == task]
        X_task = task_data[measures].values
        
        print(f"\nSpectral Radius Direction Analysis for {task}")
        print("-" * 40)
        y_sr = task_data["sr_direction_local"].values
        sr_acc, sr_pred, svm_sr = train_and_evaluate(X_task, y_sr)
        print(f"Mean Cross-Validated Accuracy (SR): {sr_acc:.2%}")
        
        sr_labels = sorted(task_data["sr_direction_local"].unique())
        print("\nClassification Report (SR):")
        print(classification_report(y_sr, sr_pred, labels=sr_labels, zero_division=0))
        
        plot_confusion_matrix(y_sr, sr_pred, sr_labels, 
                            f"CM: SR Direction - {task} (Local Target)",
                            parameter=f'sr_{task}', storing=storing, experiment=experiment)
        
        plot_f1_matrix(y_sr, sr_pred, sr_labels,
                      f"F1: SR Direction - {task} (Local Target)",
                      parameter=f'sr_{task}', storing=storing, experiment=experiment)
        
        print(f"\nScale Direction Analysis for {task}")
        print("-" * 40)
        y_scale = task_data["scale_direction_local"].values
        scale_acc, scale_pred, svm_scale = train_and_evaluate(X_task, y_scale)
        print(f"Mean Cross-Validated Accuracy (Scale): {scale_acc:.2%}")
        
        scale_labels = sorted(task_data["scale_direction_local"].unique())
        print("\nClassification Report (Scale):")
        print(classification_report(y_scale, scale_pred, labels=scale_labels, zero_division=0))
        
        plot_confusion_matrix(y_scale, scale_pred, scale_labels,
                            f"CM: Scale Direction - {task} (Local Target)",
                            parameter=f'scale_{task}', storing=storing, experiment=experiment)
        
        plot_f1_matrix(y_scale, scale_pred, scale_labels,
                      f"F1: Scale Direction - {task} (Local Target)",
                      parameter=f'scale_{task}', storing=storing, experiment=experiment)

def analyze_by_process_count(df_processed, measures, storing, experiment):
    """
    Analyze SVM performance for each process count separately
    """
    process_counts = sorted(df_processed['process_count'].unique())
    
    for p_count in process_counts:
        print(f"\n{'='*50}")
        print(f"ANALYZING PROCESS COUNT: {p_count}")
        print(f"{'='*50}")
        
        pc_data = df_processed[df_processed['process_count'] == p_count]
        X_pc = pc_data[measures].values
        
        print(f"\nSpectral Radius Direction Analysis for Process Count {p_count}")
        print("-" * 50)
        y_sr = pc_data["sr_direction_local"].values
        sr_acc, sr_pred, svm_sr = train_and_evaluate(X_pc, y_sr)
        print(f"Mean Cross-Validated Accuracy (SR): {sr_acc:.2%}")
        
        sr_labels = sorted(pc_data["sr_direction_local"].unique())
        print("\nClassification Report (SR):")
        print(classification_report(y_sr, sr_pred, labels=sr_labels, zero_division=0))
        
        plot_confusion_matrix(y_sr, sr_pred, sr_labels,
                            f"CM: SR Direction - Process Count {p_count} (Local Target)",
                            parameter=f'sr_pc{p_count}', storing=storing, experiment=experiment)
        
        plot_f1_matrix(y_sr, sr_pred, sr_labels,
                      f"F1: SR Direction - Process Count {p_count} (Local Target)",
                      parameter=f'sr_pc{p_count}', storing=storing, experiment=experiment)
        
        print(f"\nScale Direction Analysis for Process Count {p_count}")
        print("-" * 50)
        y_scale = pc_data["scale_direction_local"].values
        scale_acc, scale_pred, svm_scale = train_and_evaluate(X_pc, y_scale)
        print(f"Mean Cross-Validated Accuracy (Scale): {scale_acc:.2%}")
        
        scale_labels = sorted(pc_data["scale_direction_local"].unique())
        print("\nClassification Report (Scale):")
        print(classification_report(y_scale, scale_pred, labels=scale_labels, zero_division=0))
        
        plot_confusion_matrix(y_scale, scale_pred, scale_labels,
                            f"CM: Scale Direction - Process Count {p_count} (Local Target)",
                            parameter=f'scale_pc{p_count}', storing=storing, experiment=experiment)
        
        plot_f1_matrix(y_scale, scale_pred, scale_labels,
                      f"F1: Scale Direction - Process Count {p_count} (Local Target)",
                      parameter=f'scale_pc{p_count}', storing=storing, experiment=experiment)

def main():
    file = "../../data/RASTER_DATA_GEN.csv"
    df = pd.read_csv(file)
    print(f"Loaded data from {file} with shape: {df.shape}")
    storing =  None #"gradient_analysis_results/comprehensive"
    experiment = "comprehensive"
    
    df_full = load_data(file)
    if df_full.empty:
        return

    MEASURES = [
        "average_state_entropy", 
        "variance_activation_derivatives", 
        "participation_ratio",
        "active_information_storage", 
        "transfer_entropy", 
        "mean_correlation"
    ]

    print(f"Loaded data with shape: {df_full.shape}")
    print(f"Tasks in dataset: {sorted(df_full['task'].unique())}")
    print(f"Process counts in dataset: {sorted(df_full['process_count'].unique())}")

    df_processed = create_local_targets(df_full)
    print(f"Processed data shape: {df_processed.shape}")
    
    print(f"\nSpectral Radius Direction Distribution:")
    print(df_processed['sr_direction_local'].value_counts())
    print(f"\nScale Direction Distribution:")
    print(df_processed['scale_direction_local'].value_counts())

    X = df_processed[MEASURES].values

    y_sr = df_processed["sr_direction_local"].values
    sr_acc, sr_pred, svm_sr = train_and_evaluate(X, y_sr)
    print(f"Mean Cross-Validated Accuracy (SR): {sr_acc:.2%}")
    
    sr_labels = sorted(df_processed["sr_direction_local"].unique())
    print("\nClassification Report (SR):")
    print(classification_report(y_sr, sr_pred, labels=sr_labels, zero_division=0))
    
    plot_confusion_matrix(y_sr, sr_pred, sr_labels, 
                         "CM: SR Direction (Local Target, Overall)",
                         parameter='sr_overall', storing=storing, experiment=experiment)
    
    plot_f1_matrix(y_sr, sr_pred, sr_labels,
                   "F1: SR Direction (Local Target, Overall)",
                   parameter='sr_overall', storing=storing, experiment=experiment)

    y_scale = df_processed["scale_direction_local"].values
    scale_acc, scale_pred, svm_scale = train_and_evaluate(X, y_scale)
    print(f"\nMean Cross-Validated Accuracy (Scale): {scale_acc:.2%}")
    
    scale_labels = sorted(df_processed["scale_direction_local"].unique())
    print("\nClassification Report (Scale):")
    print(classification_report(y_scale, scale_pred, labels=scale_labels, zero_division=0))
    
    plot_confusion_matrix(y_scale, scale_pred, scale_labels,
                         "CM: Scale Direction (Local Target, Overall)",
                         parameter='scale_overall', storing=storing, experiment=experiment)
    
    plot_f1_matrix(y_scale, scale_pred, scale_labels,
                   "F1: Scale Direction (Local Target, Overall)",
                   parameter='scale_overall', storing=storing, experiment=experiment)

    analyze_by_task(df_processed, MEASURES, storing, experiment)
    analyze_by_process_count(df_processed, MEASURES, storing, experiment)
    
if __name__ == "__main__":
    main()