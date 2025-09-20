import numpy as np
# regression loss functions
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    y_true = y_true.ravel()  
    y_pred = y_pred.ravel()
    return np.mean((y_true - y_pred) ** 2)
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))
def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Mean Squared Error"""
    #print("y_test[:10]  =", y_true[:10])
    return mse(y_true, y_pred) / np.var(y_true)
def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Root Mean Squared Error"""
    return rmse(y_true, y_pred) / np.std(y_true)

# classification loss functions
def bce_loss(y_true: np.array, y_pred: np.ndarray) -> float:
    """Binary Cross Entropy"""
    return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
