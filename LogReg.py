import numpy as np
from scipy.special import expit

from Lab1 import linear_step, golden_section, gradient_descent


class NumberOfSteps:
    def __init__(self, errors, steps):
        self.errors = errors
        self.steps = steps


class Logit:
    def __init__(self, alpha, solver, max_errors=100):
        assert solver in {'gradient', 'newton'}
        self.alpha = alpha
        self.w = None
        self.solver = solver
        self.max_errors = max_errors

    @staticmethod
    def __add_feature(X):
        objects_count, _ = X.shape
        ones = np.ones((objects_count, 1))
        return np.hstack((X, ones))

    def fit(self, X, y, debug_iters=None, eps=1e-5):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        X_r = Logit.__add_feature(X)

        start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

        def Q(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            losses = np.logaddexp(0, -margins)
            return (np.sum(losses) / objects_count) + (np.sum(weights ** 2) * self.alpha / 2)

        A = np.transpose(X_r * y.reshape((objects_count, 1)))

        def Q_grad(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            b = expit(-margins)
            grad = -np.matmul(A, b) / objects_count
            return grad + self.alpha * weights

        def Q_hess(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            C = np.transpose(X_r * expit(-margins).reshape((objects_count, 1)))
            D = X_r * expit(margins).reshape((objects_count, 1))
            hess = np.matmul(C, D) / objects_count
            return hess + self.alpha * np.eye(features_count + 1)

        if self.solver == 'gradient':
            trace = gradient_descent(Q, Q_grad, features_count, linear_step(golden_section), eps)
            self.w = trace["answer"]
            return NumberOfSteps(0, len(trace))
        else:
            errors = 0
            while True:
                try:
                    if errors >= self.max_errors:
                        self.w = start_w
                        return NumberOfSteps(errors, -1)
                    else:
                        trace = newton(Q, Q_grad, Q_hess, start_w, 'delta', eps=eps, cho=True)
                        self.w = trace[-1]
                        return NumberOfSteps(errors, len(trace))
                except ArithmeticError:
                    errors += 1
                    start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

    def predict(self, X):
        X_r = Logit.__add_feature(X)
        return np.sign(np.matmul(X_r, self.w)).astype(int)
