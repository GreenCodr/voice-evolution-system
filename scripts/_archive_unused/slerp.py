# scripts/slerp.py
import numpy as np


def slerp(v0, v1, alpha: float):
    """
    Spherical Linear Interpolation between two vectors.

    v0, v1 : np.ndarray (same shape)
    alpha  : float in [0, 1]
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-5:
        # vectors are almost identical
        return v0

    sin_theta = np.sin(theta)

    w0 = np.sin((1 - alpha) * theta) / sin_theta
    w1 = np.sin(alpha * theta) / sin_theta

    return w0 * v0 + w1 * v1