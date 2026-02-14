import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid(z) as 1/(1+exp(-z))"""
    #cutoff really large values
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))

def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] == 0:
        return 0.0
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    total_error = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    loss = float(total_error / (-1.0 * y_true.shape[0]))
    return loss

def compute_gradient():
    return

def predict_classes():
    return