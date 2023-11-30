from dataclasses import dataclass

import numpy as np
import ioh
from sklearn.decomposition import PCA
from scipy.linalg import cholesky


from ..algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


def asebo_compute_grads(x, loss_fn, U, alpha, sigma, min_samples=10, threshold=0.995, default_pop_size=50):
    pca_fail = False
    dims = len(x)
    try:
        pca = PCA()
        pca_fit = pca.fit(U)
        var_exp = pca_fit.explained_variance_ratio_
        var_exp = np.cumsum(var_exp)
        n_samples = np.argmax(var_exp > threshold) + 1
        if n_samples < min_samples:
            n_samples = min_samples
        # n_samples = params['num_sensings']
        U = pca_fit.components_[:n_samples]
        UUT = np.matmul(U.T, U)
        U_ort = pca_fit.components_[n_samples:]
        UUT_ort = np.matmul(U_ort.T, U_ort)
    except np.linalg.LinAlgError:
        UUT = np.zeros([dims, dims])
        n_samples = default_pop_size
        pca_fail = True

    np.random.seed(None)
    cov = (alpha / dims) * np.eye(dims) + ((1 - alpha) / n_samples) * UUT
    # cov *= params['sigma']
    mu = np.repeat(0, dims)
    # A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.zeros((n_samples, dims))
    try:
        l = cholesky(cov, check_finite=False, overwrite_a=True)
        for i in range(n_samples):
            try:
                A[i] = np.zeros(dims) + l.dot(np.random.standard_normal(dims))
            except np.linalg.LinAlgError:
                A[i] = np.random.randn(dims)
    except np.linalg.LinAlgError:
        for i in range(n_samples):
            A[i] = np.random.randn(dims)
    A /= np.linalg.norm(A, axis=-1)[:, np.newaxis]
    A *= sigma

    m = []
    for i in range(n_samples):
        m.append(loss_fn(x + A[i]) - loss_fn(x - A[i]))
    g = np.zeros(dims)
    for i in range(n_samples):
        eps = A[i, :]
        g += eps * m[i]
    g /= (2 * (sigma ** 2) * n_samples)

    if not pca_fail:
        # params['alpha'] = np.linalg.norm(np.dot(g, UUT_ort))/np.linalg.norm(np.dot(g, UUT))
        alpha = np.linalg.norm(np.dot(g, UUT)) / np.linalg.norm(np.dot(g, UUT_ort))
    else:
        alpha = 1.
    return g, n_samples, alpha

def es_compute_grads(x, loss_fn, sigma=0.01, pop_size=10):
    grad = 0
    for i in range(pop_size):
        noise = sigma / np.sqrt(len(x)) * np.random.randn(1, len(x))
        noise = noise.reshape(x.shape)
        grad += noise * (loss_fn(x + noise) - loss_fn(x - noise))
    g_hat = grad / (2 * pop_size * sigma ** 2)
    return g_hat


@dataclass
class ASEBO(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lr: float = 0.2
    sigma: float = 0.01
    lambda_: int = 10
    k: int = 1
    decay: int = 0.95

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        x_prime = np.zeros(n)
        f_prime = problem(x_prime)

        alpha, U = 0., 0
        total_sample, current_iter = 0, 0
        try:
            while not self.should_terminate(problem, self.lambda_):
                if current_iter < self.k:
                    g_hat = es_compute_grads(x_prime, problem, pop_size=self.lambda_, sigma=self.sigma)
                    n_sample = self.lambda_
                else:
                    g_hat, n_sample, alpha = asebo_compute_grads(x_prime, 
                                                                 problem, U, sigma=self.sigma, 
                                                                 alpha=alpha, default_pop_size=self.lambda_)
                x_prime -= self.lr * g_hat
                f_prime = problem(x_prime)
                if current_iter == 0:
                    U = np.dot(g_hat[:, None], g_hat[None, :])
                else:
                    U = self.decay * U + (1-self.decay) * np.dot(g_hat[:, None], g_hat[None, :])
                total_sample += n_sample * 2
                current_iter += 1

        except KeyboardInterrupt:
            pass

        return x_prime, f_prime
