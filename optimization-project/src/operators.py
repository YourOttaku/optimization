import numpy as np
from scipy.fft import dctn, idctn


def grad_xy(u):
    """прямі різниці, дві таблиці"""
    gx = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def divergence(px, py):
    """спряжений оп(-) до градієнта"""
    d = np.zeros_like(px)
    d[:, 0] += px[:, 0]
    d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    d[:, -1] -= px[:, -2]
    d[0, :] += py[0, :]
    d[1:-1, :] += py[1:-1, :] - py[:-2, :]
    d[-1, :] -= py[-2, :]
    return d


def laplacian(u):
    """оператор лапласа"""
    return divergence(*grad_xy(u))


def energy(I, noisy, lam):
    """E(I) = ½‖I − Iшумне‖² + (λ/2)‖∇I‖²."""
    gx, gy = grad_xy(I)
    return 0.5 * float(np.sum((I - noisy) ** 2)) + \
           0.5 * lam * float(np.sum(gx * gx + gy * gy))


def energy_grad(I, noisy, lam):
    """∇E(I) = (I − Iшумне) − λ·ΔI."""
    return (I - noisy) - lam * laplacian(I)


def tikhonov_exact(noisy, lam):
    """точний мінімум енергії тихонова через косинуси"""
    h, w = noisy.shape
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    mu = (2 - 2 * np.cos(np.pi * i / h)) + (2 - 2 * np.cos(np.pi * j / w))
    spectrum = dctn(noisy, type=2, norm="ortho")
    return idctn(spectrum / (1 + lam * mu), type=2, norm="ortho")