#!/bin/bash
# Reservoir Computing Framework - Run All Experiments
# This script runs all experimental scripts in the correct order

set -e  # Exit on any error

echo "🎯 Starting Reservoir Computing Framework Experiments"
echo "📅 $(date)"
echo "📁 Working directory: $(pwd)"

# Change to scripts directory
cd "$(dirname "$0")"

# Create output directories
mkdir -p ../../data
mkdir -p ../../figures

echo "📊 Phase 1: Data Generation Scripts"
echo "================================="

# Phase 1: Basic performance studies
python3 1_all_tasks_mse_spectral_radius_BINARY.py
python3 1_all_tasks_mse_spectral_radius_ou.py
python3 1_all_tasks_mse_spectral_radius_rp.py

# Phase 2: Scaling studies
python3 2_all_tasks_mse_scaling_BINARY.py
python3 2_all_tasks_mse_scaling_ou.py
python3 2_all_tasks_mse_scaling_rp.py

# Phase 2b: Sparse inputs
python3 2_all_tasks_mse_spectral_radius_BINARY_sparse_inputs.py
python3 2_all_tasks_mse_spectral_radius_ou_sparse_inputs.py
python3 2_all_tasks_mse_spectral_radius_rp_sparse_inputs.py

# Phase 3: Measures studies
python3 3_measures_vs_spectral_radius_BINARY.py
python3 3_measures_vs_spectral_radius_OU.py
python3 3_measures_vs_spectral_radius_rp.py

# Phase 4: Measure correlations
python3 4_measure_vs_measure_BINARY.py
python3 4_measure_vs_measure_rp.py

# Phase 5: Combined analysis
python3 5_measure_nmse_data_gen.py

# Phase 7: Hyperparameter data
python3 7_raster_data_gen.py
python3 7_raster_data_gen_global.py

echo ""
echo "📈 Phase 2: Analysis & Visualization Scripts"
echo "==========================================="

# Analysis scripts
python3 5_pearson_measure_nmse.py
python3 5_spearman_measure_nmse.py
python3 6_nmse_linear_regression.py
python3 6_nmse_svr.py
python3 7_hyperparameter_prediction_model_global.py
python3 7_hyperparameter_prediction_model_local.py
python3 8_iterative_gradient_binary.py
python3 8_iterative_gradient_delay.py
python3 8_iterative_gradient_memory_tasks.py
python3 visualize_nmse.py

echo ""
echo "🎉 All experiments completed successfully!"
echo "📁 Check ../../data/ for generated datasets"
echo "🖼️  Check ../../figures/ for generated plots"
echo "📅 Finished: $(date)"
