import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

FIGSIZE_SINGLE = (3.6, 3.2)  
CMAP = "coolwarm"
VMIN, VMAX = -1.0, 1.0

sns.set_context("paper")
sns.set_style("white")

HEATMAP_KW = dict(
    cmap=CMAP, vmin=VMIN, vmax=VMAX,
    square=True, linewidths=0.4, linecolor="white",
    annot=True, fmt=".2f", annot_kws={"fontsize": 6.5}
)

ALL_POSSIBLE_MEASURES = [
    "average_state_entropy",
    "variance_activation_derivatives",  
    "participation_ratio",
    "active_information_storage",
    "mean_correlation",
    "transfer_entropy",
]

MEASURE_ORDER = [
    "participation_ratio",
    "variance_activation_derivatives",
    "average_state_entropy",
    "active_information_storage",
    "mean_correlation",
    "transfer_entropy",
]

DISPLAY_LABELS = {
    "participation_ratio": "PR",
    "variance_activation_derivatives": "VAD",
    "average_state_entropy": "ASE",
    "active_information_storage": "AIS",
    "mean_correlation": "MC",
    "transfer_entropy": "TE",
    "nmse_error": "NMSE",
}

TARGET_COL = "nmse_error"
SPECTRAL_RANGES = {
    "Sub-critical": (0.2, 0.7),
    "Edge of Chaos": (0.7, 1.3),
    "Super-critical": (1.3, 1.8),
}

def _rename_to_caps(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c: DISPLAY_LABELS.get(c, c) for c in df.columns}
    idx_map = {i: DISPLAY_LABELS.get(i, i) for i in df.index}
    return df.rename(columns=col_map, index=idx_map)

def _finalize_ax(ax, title=""):
    if title:
        ax.set_title(title, fontsize=9, pad=4)  
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)

def _add_side_colorbar(ax, mappable, label="Correlation"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.10) 
    cb = ax.figure.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

def load_and_prepare_data(file_path: str, input_type: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "te" in df.columns and "transfer_entropy" not in df.columns:
        df.rename(columns={"te": "transfer_entropy"}, inplace=True)
    df["input_type"] = input_type
    return df

def aggregate_by_setting(df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    keys = ["spectral_radius", "scale", "process_count"]
    agg = (
        df.groupby(keys, as_index=False)[value_cols]
          .mean()
    )
    return agg

def analyze_correlations(df: pd.DataFrame, available_measures: list, input_type: str, storing: str):
    """Full (measures + NMSE) Pearson matrix in CAPS; compact sizing with side colorbar."""
    valid_cols = available_measures + [TARGET_COL]
    df_used = aggregate_by_setting(df[valid_cols + ["spectral_radius","scale","process_count"]], valid_cols)

    ordered_measures = [m for m in MEASURE_ORDER if m in available_measures]
    ordered_cols = ordered_measures + [TARGET_COL]
    corr_matrix = df_used[ordered_cols].corr(method='pearson')
    corr_caps = _rename_to_caps(corr_matrix)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)
    hm = sns.heatmap(corr_caps, **HEATMAP_KW, cbar=False)
    _add_side_colorbar(ax, hm.collections[0], label="Correlation")
    _finalize_ax(ax, title=f"Global Pearson ({input_type.upper()})")
    plt.tight_layout()
    #plt.savefig(os.path.join(storing, f"5_nmse_correlation_{input_type}.pdf"), bbox_inches="tight")
    plt.show()

def analyze_conditional_correlations_sr(df: pd.DataFrame, available_measures: list, input_type: str, task_name: str, storing: str):
    """Rows: measures; Cols: SR regimes; cells: corr(measure, NMSE)."""
    valid_cols = available_measures + [TARGET_COL]
    results = {}
    for name, (low, high) in SPECTRAL_RANGES.items():
        range_df = df[(df['spectral_radius'] >= low) & (df['spectral_radius'] < high)]
        range_df_used = aggregate_by_setting(range_df[valid_cols + ["spectral_radius","scale","process_count"]], valid_cols)
        corr_matrix = range_df_used[valid_cols].corr(method='pearson')
        results[name] = corr_matrix.get(TARGET_COL, pd.Series(np.nan)).drop(TARGET_COL, errors='ignore')

    results_df = pd.DataFrame(results)
    results_df = results_df.reindex([m for m in MEASURE_ORDER if m in available_measures])
    results_df_caps = _rename_to_caps(results_df)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)
    hm = sns.heatmap(results_df_caps, **HEATMAP_KW, cbar=False)
    _add_side_colorbar(ax, hm.collections[0], label="Pearson Correlation")
    _finalize_ax(ax, title=f"NMSE vs SR ({task_name.title()})")
    ax.set_ylabel("Measures", fontsize=8)
    ax.set_xlabel("Spectral Radius Regime", fontsize=8)
    plt.tight_layout()
    #plt.savefig(os.path.join(storing, f"5_nmse_correlation_by_sr_{task_name}.pdf"), bbox_inches="tight")
    plt.show()

def analyze_correlations_by_input_count(df: pd.DataFrame, available_measures: list, input_type: str, task_name: str, storing: str):
    """Rows: measures; Cols: input counts; cells: corr(measure, NMSE)."""
    valid_cols = available_measures + [TARGET_COL]
    results = {}
    for count in sorted(df['process_count'].unique()):
        sub = df[df['process_count'] == count]
        sub_used = aggregate_by_setting(sub[valid_cols + ["spectral_radius","scale","process_count"]], valid_cols)
        corr_matrix = sub_used[valid_cols].corr(method='pearson')
        results[f"Inputs: {count}"] = corr_matrix.get(TARGET_COL, pd.Series(np.nan)).drop(TARGET_COL, errors='ignore')

    results_df = pd.DataFrame(results)
    results_df = results_df.reindex([m for m in MEASURE_ORDER if m in available_measures])
    results_df_caps = _rename_to_caps(results_df)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)
    hm = sns.heatmap(results_df_caps, **HEATMAP_KW, cbar=False)
    _add_side_colorbar(ax, hm.collections[0], label="Pearson Correlation")
    _finalize_ax(ax, title=f"NMSE vs Inputs ({task_name.title()})")
    ax.set_ylabel("Measures", fontsize=8)
    ax.set_xlabel("Number of Input Processes", fontsize=8)
    plt.tight_layout()
    #plt.savefig(os.path.join(storing, f"5_nmse_correlation_by_input_count_{task_name}.pdf"), bbox_inches="tight")
    plt.show()

def analyze_correlations_by_scale(df: pd.DataFrame, available_measures: list, input_type: str, task_name: str, storing: str):
    """Rows: measures; Cols: scales; cells: corr(measure, NMSE)."""
    valid_cols = available_measures + [TARGET_COL]
    results = {}
    for sc in sorted(df['scale'].unique()):
        sub = df[df['scale'] == sc]
        sub_used = aggregate_by_setting(sub[valid_cols + ["spectral_radius","scale","process_count"]], valid_cols)
        corr_matrix = sub_used[valid_cols].corr(method='pearson')
        results[f"Scale: {sc}"] = corr_matrix.get(TARGET_COL, pd.Series(np.nan)).drop(TARGET_COL, errors='ignore')

    results_df = pd.DataFrame(results)
    results_df = results_df.reindex([m for m in MEASURE_ORDER if m in available_measures])
    results_df_caps = _rename_to_caps(results_df)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)
    hm = sns.heatmap(results_df_caps, **HEATMAP_KW, cbar=False)
    _add_side_colorbar(ax, hm.collections[0], label="Pearson Correlation")
    _finalize_ax(ax, title=f"NMSE vs Scale ({task_name.title()})")
    ax.set_ylabel("Measures", fontsize=8)
    ax.set_xlabel("Input Scale", fontsize=8)
    plt.tight_layout()
    #plt.savefig(os.path.join(storing, f"5_nmse_correlation_by_scale_{task_name}.pdf"), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    home_dir = "../../data"
    storing = "../../figures"
    data_files = {
        "rp": "ALL_MEASURES_NMSE_CORR_DATA.csv",
    }
    os.makedirs(storing, exist_ok=True)
    
    for input_type, file_name in data_files.items():
        file_path = os.path.join(home_dir, file_name)
        df_full = load_and_prepare_data(file_path, input_type)
        available_measures = [m for m in ALL_POSSIBLE_MEASURES if m in df_full.columns]

        for task in df_full['task'].unique():
            task_df = df_full[df_full['task'] == task].copy()

            # 1) Full matrix
            analyze_correlations(task_df, available_measures, input_type, storing)

            # 2) Conditioned matrices (uniform size, compact)
            analyze_conditional_correlations_sr(
                df=task_df,
                available_measures=available_measures,
                input_type=input_type,
                task_name=task,
                storing=storing
            )
            analyze_correlations_by_input_count(
                df=task_df,
                available_measures=available_measures,
                input_type=input_type,
                task_name=task,
                storing=storing
            )
            analyze_correlations_by_scale(
                df=task_df,
                available_measures=available_measures,
                input_type=input_type,
                task_name=task,
                storing=storing
            )
