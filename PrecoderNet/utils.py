import numpy as np

def H(arr: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        arr (np.ndarray): input

    Returns:
        np.ndarray: Transpose Conjugate (Hermitian)
    """
    return np.conjugate(arr.T)