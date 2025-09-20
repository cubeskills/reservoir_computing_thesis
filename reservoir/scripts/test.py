import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from utils.simulation_tools import build_reservoir
from utils.metrics import nmse
from utils.sequence_generator import parity_task

N_STEPS = 5000
SPECTRAL_RADIUS = 0.2
SCALE = 0.7
PROCESS_COUNTS = [1, 3, 5]

def evaluate_parity(process_count, seed=42):
    X, y = parity_task(n_steps=N_STEPS, history_length=2, input_dim=process_count, seed=seed)
    n_train = int(0.8 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    reservoir = build_reservoir(
        input_dim=process_count,
        reservoir_size=100,
        spectral_radius=SPECTRAL_RADIUS,
        connectivity=0.1,
        leak_rate=1.0,
        scale=SCALE,
        seed=seed
    )

    reservoir.train(X_train.copy(), y_train.copy())
    y_pred = reservoir.predict(X_test.copy())
    return nmse(y_test, y_pred)

def main():
    for pc in PROCESS_COUNTS:
        score = evaluate_parity(pc)
        print(f"Process count {pc}: NMSE = {score:.4f}")

if __name__ == "__main__":
    main()
