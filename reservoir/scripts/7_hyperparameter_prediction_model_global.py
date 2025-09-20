import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cycler import cycler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import shap
from helpers_gradient_prediction import (

    load_data,
    train_and_evaluate,
    grid_search_and_evaluate,
    plot_confusion_matrix,
    create_targets_percentile_only,
    plot_errors_on_hp_grid,
    prepare_data_for_plotting,
    plot_heatmap_landscape,
    get_and_plot_feature_importance
    
)

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

def analyze_feature_importance_with_shap(model_pipeline, X_train, feature_names, model_title=""):
    print(f"\n--- Running SHAP Analysis for {model_title} ---")
    
    X_train_summary = shap.kmeans(X_train, 50)

    explainer = shap.KernelExplainer(model_pipeline.predict_proba, X_train_summary)

    print("Calculating SHAP values... (this may take a few minutes)")
    shap_values = explainer.shap_values(X_train)

    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values, X_train, 
        feature_names=feature_names,
        class_names=model_pipeline.classes_,
        show=False,
        plot_size=None
    )
    plt.title(f"SHAP Feature Importance for {model_title}")
    plt.tight_layout()
    plt.show()

def main():
    file = "../../data/RASTER_DATA_GEN.csv"
    output_dir = "../../figures"

    df_full = load_data(file)
    df_full = df_full[df_full['task'].isin(["delay","delay_polynomial","count_ones_in_window","narma"])]
    # shape
    print(f"Data shape: {df_full.shape}")
    df_processed = create_targets_percentile_only(df_full, percentile=0.05)
    
    MEASURES = [
        "average_state_entropy", "variance_activation_derivatives", "participation_ratio",
        "active_information_storage", "transfer_entropy", "mean_correlation"]
    
    X = df_processed[MEASURES].values
    ################################### sr
    y_sr = df_processed["sr_direction_global"].values

    sr_acc, sr_pred, svm = train_and_evaluate(X, y_sr)
    print(f"Mean Cross-Validated Accuracy on Full Dataset (SR): {sr_acc:.2%}")
    sr_labels = sorted(df_processed["sr_direction_global"].unique())

    print("\nGrid Search for SR Direction Model")
    y_sr = df_processed["sr_direction_global"].values
    
    #sr_score, sr_pred = grid_search_and_evaluate(X, y_sr)
    
    print(classification_report(y_sr, sr_pred, labels=sr_labels, zero_division=0))
    plot_confusion_matrix(y_sr, sr_pred, sr_labels, "CM: SR Direction (Hybrid Target, Full Dataset)")
    plot_errors_on_hp_grid(df_processed, sr_pred, task="Delay, Delay Polynomial, Narma, Counting Task", storing=output_dir, parameter='sr')
    sr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf',C=100,gamma='scale', probability=True))
    ])
    sr_pipeline.fit(X, y_sr)
    #get_and_plot_feature_importance(sr_pipeline, X, y_sr, MEASURES, "SR Direction")
    #analyze_feature_importance_with_shap(sr_pipeline, X, MEASURES, "SR Direction")
    
    ################################## scale
    y_scale = df_processed["scale_direction_global"].values
    scale_acc, scale_pred, svm = train_and_evaluate(X, y_scale)
    print(f"Mean Cross-Validated Accuracy on Full Dataset (Scale): {scale_acc:.2%}")
    scale_labels = sorted(df_processed["scale_direction_global"].unique())

    print("\nGrid Search for Scale Direction Model")
    y_scale = df_processed["scale_direction_global"].values
    #scale_score, scale_pred = grid_search_and_evaluate(X, y_scale)
    
    print(classification_report(y_scale, scale_pred, labels=scale_labels, zero_division=0))
    plot_confusion_matrix(y_scale, scale_pred, scale_labels, "CM: Scale Direction (Hybrid Target, Full Dataset)")
    plot_errors_on_hp_grid(df_processed, scale_pred, task="Delay, Delay Polynomial, Narma, Counting Task", storing=output_dir, parameter='scale')
    
    scale_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=100,gamma='scale', probability=True)) # 
    ])

    #scale_pipeline.fit(X, y_scale)
    #get_and_plot_feature_importance(scale_pipeline, X, y_scale, MEASURES, "Scale Direction")
    #analyze_feature_importance_with_shap(scale_pipeline, X, MEASURES, fontsize=16, "Scale Direction")

    """
    for task in df_processed['task'].unique():
        for p_count in sorted(df_processed['process_count'].unique()):
        
            print(f"--- Generating plots for Task: {task}, Process Count: {p_count} ---")
            
            pivoted_data = prepare_data_for_plotting(df_processed, task, p_count)
            
            if pivoted_data is not None and not pivoted_data.empty:
                plot_heatmap_landscape(pivoted_data, task, p_count)
    """
if __name__ == "__main__":
    main()
