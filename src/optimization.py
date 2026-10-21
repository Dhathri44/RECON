import numpy as np


def objective(x, y, observed_mask, lam=0.1):
    data_error = observed_mask * (x - y)

    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]

    data_term = 0.5 * np.sum(data_error ** 2)

    smoothness_term = 0.5 * lam * (
        np.sum(dx ** 2) + np.sum(dy ** 2)
    )

    return data_term + smoothness_term


def gradient(x, y, observed_mask, lam=0.1):
    grad = observed_mask * (x - y)

    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]

    grad[:, :-1] -= lam * dx
    grad[:, 1:] += lam * dx

    grad[:-1, :] -= lam * dy
    grad[1:, :] += lam * dy

    return grad


def gradient_descent(
    y,
    observed_mask,
    learning_rate=0.1,
    iterations=200,
    lam=0.1
):
    x = y.astype(np.float64).copy()

    history = []

    for iteration in range(iterations):
        grad = gradient(
            x,
            y,
            observed_mask,
            lam
        )

        x -= learning_rate * grad

        value = objective(
            x,
            y,
            observed_mask,
            lam
        )

        history.append(value)

    return x, history