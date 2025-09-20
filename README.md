# Reservoir Computing Framework

A framework for Reservoir Computing systems.

## Overview

This project presents a principled, efficient, and automated method for tuning reservoir computers using intrinsic dynamics measures rather than expensive task-specific searches. 

## The Problem We Solve

### The "Black-Box Art" of Reservoir Tuning
- Optimizing key hyperparameters (spectral radius, input scaling) typically requires exhaustive, task-specific searches
- Traditional methods are computationally expensive and impractical for real-time applications
- Physical reservoir systems (photonic circuits, biological substrates) don't allow direct parameter setting

### The Physical Reservoir Challenge
In physical systems, you can't simply set a spectral radius - you can only influence global properties like:
- Temperature
- Voltage  
- Physical constraints

This makes traditional optimization methods **impractical**.

## Our Solution: Closed-Loop Unsupervised Control

Instead of using expensive task performance measurements, our controller uses **intrinsic, task-agnostic measures** derived from the reservoir's internal dynamics.

### Key Innovation
A **Support Vector Machine (SVM)** learns the complex, nonlinear relationship between internal dynamics and task performance, enabling real-time hyperparameter optimization.

## Intrinsic Measures

Our framework leverages measures from **information theory** and **state-space analysis**:

### Information Theoretic Measures
- **Average State Entropy** - captures dynamical complexity
- **Active Information Storage** - quantifies memory capacity
- **Transfer Entropy** - measures information transfer between components

### State-Space Complexity Measures  

- **Participation Ratio** - measures effective dimensionality of state space
- **Variance of Activation Derivatives** - quantifies nonlinearity
- **Mean Correlation** - measures synchronization

## How the Controller Works

```
1. OBSERVE → Monitor reservoir's internal dynamics through intrinsic measures
2. PREDICT → SVM classifier predicts optimal hyperparameter directions  
3. ACT → Update hyperparameters (increase/decrease/stay)
4. REPEAT → Iterative optimization without expensive retraining
```

## Key Findings & Contributions

### ✅ Nonlinearity is Essential
- The relationship between internal measures and task performance is **highly nonlinear**
- Nonlinear models (SVMs) significantly outperform linear approaches

### ✅ Accurate Guidance
- High classification accuracy for hyperparameter direction prediction
- Successful optimization across diverse task types

### ✅ Transferable Intelligence
- Controllers trained on simple tasks (e.g., memory tasks) successfully optimize for complex, unseen tasks (e.g., NARMA10)
- **No re-optimization needed** for new problems

### ✅ Practical Viability
- Real-time steering from poor to optimal performance regimes
- Computationally efficient - no expensive grid searches required

### ✅ Hardware-Ready
- Only requires **observable data** and **external controls**
- Interesting applciation for physical reservoirs where internal weights are inaccessible are possible

## 📊 Project Structure

```
reservoir_framework/
├── data/                          # Experimental data and results
├── figures/                       # Generated plots and visualizations  
├── reservoir/
│   ├── core/                      # Core reservoir implementations
│   │   ├── reservoir.py           # Main reservoir class
│   │   ├── input_mapping.py       # Input mapping
│   │   └── readout.py            # Output layer training
│   ├── training/                  # Training utilities (combining the core scripts)
│   │   └── trainer.py            # Reservoir trainer
│   ├── utils/                     # Utility functions
│   │   ├── measures.py           # Intrinsic dynamics measures
|   |   |── metrics.py              # nmse implementation (otherwise useless)
│   │   ├── sequence_generator.py # Task generators
│   │   └── simulation_tools.py   # Simulation helpers
│   └── scripts/                   # Experimental scripts
├── requirements.txt               # Python dependencies
├── setup.py                      # Package installation
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd reservoir_framework

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Basic Usage

```python
from reservoir.core.reservoir import Reservoir
from reservoir.training.trainer import ReservoirTrainer
from reservoir.utils.measures import Measures

# Create and configure reservoir
trainer = build_reservoir(
    input_dim=3,
    reservoir_size=100,
    spectral_radius=1.2,
    scale=0.5,
    seed=42
)

# Train on task
trainer.train(X_train, y_train)

# Make predictions  
y_pred = trainer.predict(X_test)

# Compute intrinsic measures
measures = Measures(trainer.current_history, trainer.states)
ase = measures.average_state_entropy()
...
...
...

```

## Experimental Scripts

The `scripts/` directory contains various experimental studies:

- **`1_*`** - Basic task performance vs. spectral radius studies
- **`2_*`** - Scaling parameter investigations  
- **`3_*`** - Intrinsic measures vs. hyperparameters
- **`4_*`** - Measure correlation analyses
- **`5_*`** - Measure-performance relationship studies
- **`6_*`** - Machine learning prediction models
- **`7_*`** - Hyperparameter prediction and optimization
- **`8_*`** - Iterative gradient-based controllers

## Results & Impact

### Performance Improvements
- **Eliminates exhaustive grid searches** - reduces optimization time by orders of magnitude
- **Real-time adaptation** - enables online hyperparameter tuning
- **Task-agnostic optimization** - one controller works across multiple problem domains

### Computational Benefits
- **Reduced training time** for reservoir optimization
- **Automated hyperparameter selection** without domain expertise
- **Online adaptation** to changing environmental conditions

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{dinkler2025unsupervised,
  title={Intrinsic Reservoir Metrics for evaluating Echo-State Networks and Computational Performance},
  author={Paul Dinkler},
  year={2025},
  school={[University of Göttingen]},
  type={Bachelor's Thesis}
}
```

