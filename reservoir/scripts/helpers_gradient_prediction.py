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


def _format_colorbar(cbar, label):
    """Match colorbar typography to the theme."""
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])


def load_data(file):
    if not os.path.exists(file):
        print(f"Warning: Data file '{file}' not found. Cannot proceed.")
        return pd.DataFrame()
    return pd.read_csv(file)


def train_and_evaluate(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel='rbf', C=100, class_weight=None, gamma='scale')
    acc_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring="accuracy")
    y_pred = cross_val_predict(svm, X_scaled, y, cv=5)
    return acc_scores.mean(), y_pred, svm


def get_and_plot_feature_importance(model, X, y, feature_names, model_name=""):
    print(f"Calculating permutation importance for {model_name}...")
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False)

    print(f"\nFeature Importance for {model_name}")
    print(importance_df)

    fig, ax = plt.subplots()
    colors = sns.color_palette('viridis', n_colors=len(importance_df))
    ax.barh(
        y=importance_df['feature'],
        width=importance_df['importance_mean'],
        xerr=importance_df['importance_std'],
        color=colors,
        capsize=5
    )
    ax.invert_yaxis()
    ax.set_title(f'Permutation Feature Importance for {model_name}')
    ax.set_xlabel('Importance (Mean Decrease in Score)')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()


def grid_search_and_evaluate(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf'))
    ])
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 1, 0.1, 0.01],
        'svm__class_weight': ['balanced', None]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1-weighted score: {grid_search.best_score_:.2%}")
    best_model = grid_search.best_estimator_
    y_pred = cross_val_predict(best_model, X, y, cv=5)
    
    return grid_search.best_score_, y_pred


def plot_confusion_matrix(y_true, y_pred, class_labels, title="Confusion Matrix", storing=None, parameter=None, experiment=None):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots()
    hm = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        #annot_kws={"color": "black"},
        cbar_kws={"label": "Count"}
    )
    ax.set_title(title)
    ax.set_ylabel('Actual Direction')
    ax.set_xlabel('Predicted Direction')

    cbar = hm.collections[0].colorbar
    _format_colorbar(cbar, "Count")

    plt.tight_layout()
    if storing:
        os.makedirs(storing, exist_ok=True)
        filename = f"{experiment}_experiment_svm_confusion_matrix_{parameter}_local.png"
        plt.savefig(os.path.join(storing, filename))
        print(f"Saved confusion matrix plot to {os.path.join(storing, filename)}")
    plt.show()
    plt.close()


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


def plot_errors_on_hp_grid(df, y_pred, task, storing=None, parameter='sr', experiment='first'):
    """
    Plot SVM classification accuracy over spectral_radius × scale grid.
    Uses unified thesis formatting; ensures black annotations and 2-decimal precision.
    """
    df_plot = df.copy()
    df_plot['predicted_direction'] = y_pred
    true_direction_col = f'{parameter}_direction_global'
    df_plot['is_correct'] = (df_plot[true_direction_col] == df_plot['predicted_direction'])

    accuracy_grid = (
        df_plot.groupby(['spectral_radius', 'scale'])['is_correct']
        .mean()
        .unstack()
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    if accuracy_grid.empty:
        print(f"[INFO] No data for task={task}, parameter={parameter}")
        return

    fig, ax = plt.subplots()
    hm = sns.heatmap(
        accuracy_grid,
        ax=ax,
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",                    
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Accuracy"}
    )

    if parameter == 'sr':
        p = 'Spectral Radius'
    elif parameter == 'scale':
        p = 'Scale'
    else:
        p = parameter

    ax.set_title(f"SVM Accuracy: {p} Direction Classification\nTask: \"{task}\"")
    ax.set_xlabel("Scale (log-grid)")
    ax.set_ylabel("Spectral Radius")

    ax.set_xticklabels([f"{val:.3f}" for val in accuracy_grid.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{val:.2f}" for val in accuracy_grid.index], rotation=0)

    
    cbar = hm.collections[0].colorbar
    _format_colorbar(cbar, "Accuracy")

    plt.tight_layout()
    if storing is not None:
        os.makedirs(storing, exist_ok=True)
        filename = f"7_{experiment}_experiment_svm_accuracy_grid_{task}_{parameter}.pdf"
        out_path = os.path.join(storing, filename)
        plt.savefig(out_path)
        print(f"Saved accuracy grid plot to {out_path}")

    plt.show()
    plt.close()


def prepare_data_for_plotting(df, task_name, p_count):
    condition_df = df[(df['task'] == task_name) & (df['process_count'] == p_count)]
    error_landscape = condition_df.groupby(['scale', 'spectral_radius'])['base_nmse'].mean().reset_index()
    pivoted_df = error_landscape.pivot(index='scale', columns='spectral_radius', values='base_nmse')
    pivoted_df = pivoted_df.sort_index().sort_index(axis=1)
    return pivoted_df


def create_local_targets(df):
    """
    Maps the direction strings from the data generation script to standardized 
    'increase', 'decrease', or 'stay' labels for classification.
    """
    df_processed = df.copy()
    sr_map = {
        'sr_increase': 'increase',
        'sr_decrease': 'decrease',
        'base': 'stay'
    }
    df_processed['sr_direction_local'] = df_processed['best_sr_direction'].map(sr_map)
    
    scale_map = {
        'scale_increase': 'increase',
        'scale_decrease': 'decrease',
        'base': 'stay'
    }
    df_processed['scale_direction_local'] = df_processed['best_scale_direction'].map(scale_map)
    
    df_processed.dropna(subset=['sr_direction_local', 'scale_direction_local'], inplace=True)
    return df_processed


def plot_heatmap_landscape(pivoted_df, task_name, p_count, storing):
    fig, ax = plt.subplots()

    norm = LogNorm(vmin=pivoted_df.min().min() + 1e-9, vmax=pivoted_df.max().max())

    hm = sns.heatmap(
        pivoted_df,
        ax=ax,
        cmap='viridis_r',  
        norm=norm,
        cbar_kws={'label': 'Mean NMSE (log scale)'}
    )

    ax.set_title(f'NMSE Landscape: {task_name} (Input Processes: {p_count})')
    ax.set_xlabel('Spectral Radius')
    ax.set_ylabel('Input Scale')

    y_labels = [f"{val:.3f}" for val in pivoted_df.index]
    ax.set_yticklabels(y_labels)

    cbar = hm.collections[0].colorbar
    _format_colorbar(cbar, 'Mean NMSE (log scale)')

    plt.tight_layout()
    plt.show()

    if storing is not None:
        os.makedirs(storing, exist_ok=True)
        filename = f"nmse_landscape_{task_name}_process_count_{p_count}.png"
        plt.savefig(os.path.join(storing, filename))
        print(f"Saved heatmap to {os.path.join(storing, filename)}")


def process_data(df, target_col=None):
    df.loc[df['relative_improvement'] < 1e-2, target_col] = "None"
    return df


def train_test_svm(X, y, kernel='rbf', C=1.0, gamma='scale', feature_importance=False):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel=kernel, C=C, class_weight='balanced', gamma=gamma)
    cv_scores = cross_val_score(svm, X_scaled, y, cv=5)
    mean_cv_acc = cv_scores.mean()
    svm.fit(X_scaled, y)
    if feature_importance:
        result = permutation_importance(svm, X_scaled, y, n_repeats=10, random_state=42)
        feature_importances = result.importances_mean
        return svm, mean_cv_acc, feature_importances
    else:
        return svm, mean_cv_acc


def train(X, y, kernel='rbf', C=1.0, gamma='scale'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel=kernel, C=C, class_weight=None, gamma=gamma)
    svm.fit(X_scaled, y)
    return svm


def plot_feature_importance(feature_importances, feature_names, title=None):
    feature_importances = np.array(feature_importances)
    if feature_importances.ndim == 1:
        feature_importances = feature_importances.reshape(1, -1)

    fig, ax = plt.subplots()
    hm = sns.heatmap(
        feature_importances, annot=True, fmt=".2f", cmap='viridis_r',
        xticklabels=feature_names, yticklabels=['Importance'],
        annot_kws={"color": "black"}
    )
    ax.set_title(title if title else "Feature Importance Heatmap")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")

    cbar = hm.collections[0].colorbar
    _format_colorbar(cbar, "Value")

    plt.tight_layout()
    plt.show()
