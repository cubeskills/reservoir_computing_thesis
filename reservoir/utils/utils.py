import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(title_process, process, predictions, n_train, n_samples, y_test):
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(2, 1, 1)
    plt.title(title_process)
    plt.plot(process, label="Full OU Process")
    plt.axvline(n_train, color='r', linestyle='--', label="Train/Test Split")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title("ESN Prediction on Test Set")
    time_axis_test = np.arange(n_train, n_samples) 
    plt.plot(time_axis_test, y_test.flatten(), label="Actual Test Data", alpha=0.7)
    plt.plot(time_axis_test, predictions.flatten(), label="ESN Predictions", linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# generate matrix
def generate_matrix(rows: int,cols: int,sparsity: float = 0.5, spectral_radius: float = None) -> np.ndarray:
    """Generate a random matrix with given sparsity and spectral radius."""
    if rows != cols and spectral_radius is not None:
        raise ValueError("Spectral radius is not None, but matrix is not square.")
    
    # mask sets sparsity amount to 0
    mat = np.random.rand(rows, cols)
    mask = np.random.rand(rows, cols) < sparsity
    mat[mask] = 0.0

    if spectral_radius is not None:
        eigs = np.linalg.eigvals(mat)
        curr_radius = np.max(np.abs(eigs))
        if curr_radius > 0:
            mat *= spectral_radius / curr_radius

    return mat
# rescale matrix
def rescale_matrix(matrix: np.ndarray, spectral_radius: float) -> np.ndarray:
    m_max_eig = np.max(np.abs(np.linalg.eigvals(matrix)))
    return spectral_radius * matrix / (m_max_eig + 1e-12)





