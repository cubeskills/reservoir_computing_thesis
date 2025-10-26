#!/usr/bin/env python3
"""
Reservoir Computing Framework - Run All Experiments
This script runs all experimental scripts in the correct order
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Set matplotlib to non-interactive backend to prevent plots from showing
import matplotlib
matplotlib.use('Agg')

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

def run_script(script_name, description=""):
    """Run a single script and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {script_name}")
    if description:
        print(f"📝 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=3600)  # 1 hour timeout
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {script_name} ({elapsed:.1f}s)")
            if result.stdout.strip():
                print("📋 Output:", result.stdout.strip()[-200:])  # Last 200 chars
        else:
            print(f"❌ FAILED: {script_name} ({elapsed:.1f}s)")
            print("🚨 Error:", result.stderr.strip())
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {script_name} (>1 hour)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {script_name} - {e}")
        return False
    
    return True

def main():
    """Run all scripts in order."""
    print(f"🎯 Starting Reservoir Computing Framework Experiments")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directories
    os.makedirs("../../data", exist_ok=True)
    os.makedirs("../../figures", exist_ok=True)
    
    # Define execution order and descriptions
    script_order = [
        # Phase 1: Basic Task Performance Studies
        ("1_all_tasks_mse_spectral_radius_BINARY.py", "Binary tasks vs spectral radius"),
        ("1_all_tasks_mse_spectral_radius_ou.py", "OU process tasks vs spectral radius"),
        ("1_all_tasks_mse_spectral_radius_rp.py", "Random process tasks vs spectral radius"),
        
        # Phase 2: Scaling Studies
        ("2_all_tasks_mse_scaling_BINARY.py", "Binary tasks vs input scaling"),
        ("2_all_tasks_mse_scaling_ou.py", "OU process tasks vs input scaling"),
        ("2_all_tasks_mse_scaling_rp.py", "Random process tasks vs input scaling"),
        
        # Phase 2b: Sparse Input Studies
        ("2_all_tasks_mse_spectral_radius_BINARY_sparse_inputs.py", "Binary tasks with sparse inputs"),
        ("2_all_tasks_mse_spectral_radius_ou_sparse_inputs.py", "OU process with sparse inputs"),
        ("2_all_tasks_mse_spectral_radius_rp_sparse_inputs.py", "Random process with sparse inputs"),
        
        # Phase 3: Intrinsic Measures Studies
        ("3_measures_vs_spectral_radius_BINARY.py", "Measures analysis for binary tasks"),
        ("3_measures_vs_spectral_radius_OU.py", "Measures analysis for OU process"),
        ("3_measures_vs_spectral_radius_rp.py", "Measures analysis for random process"),
        
        # Phase 4: Measure Correlations
        ("4_measure_vs_measure_BINARY.py", "Measure correlations for binary tasks"),
        ("4_measure_vs_measure_rp.py", "Measure correlations for random process"),
        
        # Phase 5: Combined Measures and NMSE Analysis
        ("5_measure_nmse_data_gen.py", "Generate combined measures-NMSE dataset"),
        
        # Phase 7: Hyperparameter Prediction Data
        ("7_raster_data_gen.py", "Generate hyperparameter sweep data"),
        ("7_raster_data_gen_global.py", "Generate global hyperparameter data"),
    ]
    
    # Analysis scripts (run after data generation)
    analysis_scripts = [
        ("5_pearson_measure_nmse.py", "Pearson correlation analysis"),
        ("5_spearman_measure_nmse.py", "Spearman correlation analysis"),
        ("6_nmse_linear_regression.py", "Linear regression analysis"),
        ("6_nmse_svr.py", "Support Vector Regression analysis"),
        ("7_hyperparameter_prediction_model_global.py", "Global hyperparameter prediction"),
        ("7_hyperparameter_prediction_model_local.py", "Local hyperparameter prediction"),
        ("8_iterative_gradient_binary.py", "Iterative optimization for binary tasks"),
        ("8_iterative_gradient_delay.py", "Iterative optimization for delay tasks"),
        ("8_iterative_gradient_memory_tasks.py", "Iterative optimization for memory tasks"),
        ("visualize_nmse.py", "NMSE visualization"),
    ]
    
    total_scripts = len(script_order) + len(analysis_scripts)
    failed_scripts = []
    
    print(f"📊 Total scripts to run: {total_scripts}")
    print(f"🔄 Execution phases: Data Generation → Analysis → Visualization")
    
    # Phase 1: Data Generation Scripts
    print(f"\n🏗️  PHASE 1: DATA GENERATION ({len(script_order)} scripts)")
    for i, (script, desc) in enumerate(script_order, 1):
        print(f"\n[{i}/{len(script_order)}] {script}")
        if not run_script(script, desc):
            failed_scripts.append(script)
    
    # Phase 2: Analysis Scripts
    print(f"\n📈 PHASE 2: ANALYSIS & VISUALIZATION ({len(analysis_scripts)} scripts)")
    for i, (script, desc) in enumerate(analysis_scripts, 1):
        print(f"\n[{i}/{len(analysis_scripts)}] {script}")
        if not run_script(script, desc):
            failed_scripts.append(script)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {total_scripts - len(failed_scripts)}/{total_scripts}")
    print(f"❌ Failed: {len(failed_scripts)}/{total_scripts}")
    
    if failed_scripts:
        print(f"\n🚨 Failed scripts:")
        for script in failed_scripts:
            print(f"   - {script}")
        return 1
    else:
        print(f"\n🎉 All scripts completed successfully!")
        print(f"📁 Check ../../data/ for generated datasets")
        print(f"🖼️  Check ../../figures/ for generated plots")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
