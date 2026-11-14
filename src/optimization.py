import numpy as np



def objective(x, y, mask, lam=0.2):

 
    data_error = mask * (x - y)
    data_term = 0.5 * np.sum(data_error**2)

  
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]

  
    smoothness_term = 0.5 * lam * (
        np.sum(dx**2) + np.sum(dy**2)
    )

    return data_term + smoothness_term



def compute_gradient(x, y, mask, lam=0.2):

    grad = mask * (x - y)

    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]

    grad[:, :-1] -= lam * dx
    grad[:, 1:] += lam * dx

    grad[:-1, :] -= lam * dy
    grad[1:, :] += lam * dy

    return grad

def gradient_descent(
    y, mask, lam=0.2, iterations=500, tolerance=1e-7
):

    x = y.copy().astype(np.float64)

    
    observed_values = y[mask > 0]

    if observed_values.size > 0:
        x[mask == 0] = np.mean(observed_values)

    learning_rate = 1.0 / (1.0 + 8.0 * lam)

    history = []

    for i in range(iterations):

        grad = compute_gradient(x, y, mask, lam)

        gradient_norm = np.linalg.norm(grad) / np.sqrt(grad.size)

        if gradient_norm < tolerance:
            print("Converged at iteration:", i + 1)
            break

        x -= learning_rate * grad

        current_loss = objective(x, y, mask, lam)
        history.append(current_loss)

    return x, history



def check_gradient():

    np.random.seed(42)

    x = np.random.rand(5, 5)
    y = np.random.rand(5, 5)

    mask = np.random.randint(0, 2, (5, 5))

    lam = 0.2
    epsilon = 1e-6

    analytical = compute_gradient(x, y, mask, lam)

    numerical = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):

            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i, j] += epsilon
            x_minus[i, j] -= epsilon

            f_plus = objective(x_plus, y, mask, lam)
            f_minus = objective(x_minus, y, mask, lam)

            numerical[i, j] = (
                (f_plus - f_minus) / (2 * epsilon)
            )

    error = np.linalg.norm(analytical - numerical)

    print("Gradient verification error:", error)

    return error

