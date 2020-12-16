import torch
import numpy as np


def log_mean_exp(mtx):
    """
    Возвращает логарифм среднего по каждому столбцу от экспоненты данной матрицы.
    Вход: Tensor - матрица размера n x k.
    Выход: Tensor, вектор длины n.
    """
    m, _ = torch.max(mtx, dim=1, keepdim=True)
    outputs = m + (mtx - m).exp().mean(dim=1, keepdim=True).log()
    outputs = outputs.squeeze(1)
    return outputs


def log_likelihood(generated_set, validation_set, test_set):
    """
    Возвращает оценку логарифма правдоподобия модели GAN методом
    Парзеновского окна со стандартным нормальным ядром.
    Вход: generated_set - сэмплы из генеративной модели.
          validation_set - валидационная выборка.
          test_set - тестовая выборка.
    Выход: float - оценка логарифма правдоподобия.
    """
    M = generated_set.shape[0]
    N = validation_set.shape[0]
    D = generated_set.shape[1]
    validation_tensor = validation_set.unsqueeze(1).repeat(1, M, 1)
    test_tensor = test_set.unsqueeze(1).repeat(1, M, 1)
    generated_tensor = generated_set.unsqueeze(0).repeat(N, 1, 1)

    sigma_space = np.logspace(-4, 4, 100)
    grid_ll = np.zeros_like(sigma_space)
    for i, sigma in enumerate(sigma_space):
        mtx = - (validation_tensor - generated_tensor) ** 2 / (2 * sigma ** 2)
        mtx = mtx.sum(dim=2)
        ll_vector = log_mean_exp(mtx) - D * np.log((2 * np.pi) ** 0.5 * sigma)
        ll = ll_vector.mean().item()
        grid_ll[i] = ll

    best_sigma = sigma_space[np.argmax(grid_ll)]
    mtx = - (test_tensor - generated_tensor) ** 2 / (2 * best_sigma ** 2)
    mtx = mtx.sum(dim=2)
    ll_vector = log_mean_exp(mtx) - D * np.log((2 * np.pi) ** 0.5 * best_sigma)
    ll = ll_vector.mean().item()
    return ll
