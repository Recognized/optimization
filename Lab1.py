import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve


def dichotomy(a, b, f, eps):
    d = eps / 4

    calc = 0
    iterations = 0

    def g(x):
        nonlocal calc
        calc = calc + 1
        return f(x)

    while b - a >= eps:
        iterations = iterations + 1
        x1 = (a + b) / 2 - d
        x2 = (a + b) / 2 + d
        fx1 = g(x1)
        fx2 = g(x2)
        if fx1 < fx2:
            b = x2
        if fx2 <= fx1:
            a = x1
    return {"answer": (a + b) / 2, "iterations": iterations, "calc": calc, "name": "dichotomy"}


def golden_section(a, b, f, eps):
    calc = 0
    iterations = 0

    def g(x):
        nonlocal calc
        calc = calc + 1
        return f(x)

    x1 = a + (3 - sqrt(5)) / 2 * (b - a)
    x2 = a + (sqrt(5) - 1) / 2 * (b - a)
    fx1 = g(x1)
    fx2 = g(x2)
    while b - a >= eps:
        iterations = iterations + 1
        if fx1 < fx2:
            b = x2
            x2 = x1
            x1 = a + (3 - sqrt(5)) / 2 * (b - a)
            fx2 = fx1
            fx1 = g(x1)
        elif fx2 <= fx1:
            a = x1
            x1 = x2
            x2 = a + (sqrt(5) - 1) / 2 * (b - a)
            fx1 = fx2
            fx2 = g(x2)
    return {"answer": (a + b) / 2, "calc": calc, "iterations": iterations, "name": "golden_section"}


def number_fibonacci(n):
    return (pow((1 + sqrt(5)) / 2, n) - pow((1 - sqrt(5)) / 2, n)) / sqrt(5)


def fibonacci(a, b, f, eps):
    calc = 0
    iterations = 0

    n = 1
    while (b - a) / eps >= number_fibonacci(n + 2):
        n = n + 1

    def g(x):
        nonlocal calc
        calc = calc + 1
        return f(x)

    k = 1
    b0 = b
    a0 = a
    x1 = a + number_fibonacci(n - k + 1) / number_fibonacci(n + 2) * (b - a)
    x2 = a + number_fibonacci(n - k + 2) / number_fibonacci(n + 2) * (b - a)
    fx1 = g(x1)
    fx2 = g(x2)
    while k <= n and b - a >= eps:
        k = k + 1
        if fx1 < fx2:
            b = x2
            x2 = x1
            x1 = a + number_fibonacci(n - k + 1) / number_fibonacci(n + 2) * (b0 - a0)
            fx2 = fx1
            fx1 = g(x1)
        elif fx2 < fx1:
            a = x1
            x1 = x2
            x2 = a + number_fibonacci(n - k + 2) / number_fibonacci(n + 2) * (b0 - a0)
            fx1 = fx2
            fx2 = g(x2)
        iterations += 1
    return {"answer": (a + b) / 2, "calc": calc, "iterations": iterations, "name": "fibonacci"}


def straight(f, method, eps):
    a = 0
    b = 1
    d = 1

    iterations = 0

    f1 = f(a)
    f2 = f(b)

    calc = 2

    if f2 > f1:
        d = -d

    k1 = f1
    k2 = f2

    while f1 < f2 and k1 <= k2 or f1 > f2 and k1 >= k2:
        iterations += 1
        if d < 0:
            a = a + d
            f1 = f(a)
        else:
            b = b + d
            f2 = f(b)
        d = d * 2
        calc += 1

    m = method(a, b, f, eps)
    return {"answer": m["answer"], "calc": m["calc"] + calc, "iterations": m["iterations"] + iterations,
            "name": "straight"}


def task_1():
    def f(x):
        return 3 * x * x + 5 * x - 6

    eps = [.1, .01, .001, .0001]

    a = -1000
    b = 1000

    methods = [
        lambda s: dichotomy(a, b, f, s),
        lambda s: fibonacci(a, b, f, s),
        lambda s: golden_section(a, b, f, s),
        lambda s: straight(f, golden_section, s)
    ]

    for method in methods:
        for e in eps:
            print(method(e))


def circle(arg):
    x, y = arg
    return (y - 1) ** 2 + (x + 2) ** 2 / 4 + x + y + 1


def grad_circle(arg):
    x, y = arg
    return (x + 4) / 2, 2 * y - 1


def gradient_descent(f, df, start, search, eps):
    x = start

    x_prev = None

    trace = [x]

    iterations = 0
    calc = 0
    name = None

    while x_prev is None or np.linalg.norm(np.subtract(x, x_prev)) >= eps:
        dx = df(x)
        iterations += 1

        m = search(x, f, dx, eps)
        step = m["answer"]
        name = m["name"]
        calc += m["calc"]

        x_prev = x

        x = x + np.multiply(dx, step)

        trace.append(x)
    return {"trace": trace, "calc": calc, "iterations": iterations, "name": name}


def linear_step(method):
    def search(x, f, df, eps):
        def best(s):
            return f(x + np.multiply(s, df))

        return method(-1000, 1000, best, eps)

    return search


def constant_step(step):
    def search(x, f, df, eps):
        return {"answer": -step, "calc": 0, "name": "constant_step: %f" % step}

    return search


def armiho(e=0.1, l=1, d=0.95):
    def search(x, f, df, eps):
        t = l
        calc = 0
        while f(np.subtract(x, np.multiply(t, df))) > f(x) - e * t * np.linalg.norm(df) ** 2:
            calc = calc + 1
            t = t * d
        return {"answer": -t, "calc": calc, "name": "armiho: e=%f, l=%f, d=%f" % (e, l, d)}

    return search


def task_2():
    start = np.random.normal(loc=0., scale=5., size=2)

    steps = [
        linear_step(dichotomy),
        linear_step(lambda a, b, f, e: straight(f, golden_section, e)),
        linear_step(golden_section),
        linear_step(fibonacci),
        constant_step(0.001),
        armiho()
    ]

    for step in steps:
        m = gradient_descent(circle, grad_circle, start, step, 0.0001)
        trace = m["trace"]
        m.pop("trace")
        fig = plt.figure(figsize=(6, 5))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        x_val = np.arange(-10, 10, 0.1)
        y_val = np.arange(-10, 10, 0.1)
        X, Y = np.meshgrid(x_val, y_val)

        arg = [circle([x, y]) for x, y in zip(X, Y)]

        cp = ax.contour(X, Y, arg)
        ax.clabel(cp, inline=True, fontsize=10)
        ax.plot(*np.array(trace).T)
        ax.set_title(m["name"])
        plt.show()
        print(m)


class LogReg:
    def __init__(self, alpha, solver, max_errors=100):
        assert solver in {'gradient', 'newton'}
        self.alpha = alpha
        self.w = None
        self.solver = solver
        self.max_errors = max_errors

    def add_feature(self, X):
        objects_count, _ = X.shape
        ones = np.ones((objects_count, 1))
        return np.hstack((X, ones))

    def fit(self, X, y, eps=1e-5):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        X_r = self.add_feature(X)

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
            trace = gradient_descent(Q, Q_grad, np.random.normal(loc=0., scale=1., size=features_count + 1),
                                     linear_step(golden_section), eps=eps)["trace"]
            self.w = trace[-1]
            return 0, len(trace)
        else:
            errors = 0
            while True:
                try:
                    if errors >= self.max_errors:
                        self.w = np.random.normal(loc=0., scale=1., size=features_count + 1)
                        return errors, -1
                    else:
                        trace = newton(Q, Q_grad, Q_hess, np.random.normal(loc=0., scale=1., size=features_count + 1),
                                       eps=eps, cho=True)
                        self.w = trace[-1]
                        return errors, len(trace)
                except ArithmeticError:
                    errors += 1

    def predict(self, X):
        X_r = self.add_feature(X)
        return np.sign(np.matmul(X_r, self.w)).astype(int)


def newton(f, f_grad, f_hess, start, stop_criterion='delta', eps=1e-5, max_iters=100, cho=False):
    assert stop_criterion in {'arg', 'value', 'delta'}
    cur_arg = start
    cur_value = f(cur_arg)
    trace = [cur_arg]
    while True:
        cur_grad = f_grad(cur_arg)
        cur_hess = f_hess(cur_arg)
        if cho:
            hess_inv = cho_solve(cho_factor(cur_hess), np.eye(cur_hess.shape[0]))
        else:
            hess_inv = np.linalg.inv(cur_hess)
        cur_delta = np.matmul(cur_grad, hess_inv)
        next_arg = cur_arg - cur_delta
        next_value = f(next_arg)
        trace.append(next_arg)

        if len(trace) == max_iters:
            raise ArithmeticError()

        if (stop_criterion == 'arg' and np.linalg.norm(next_arg - cur_arg) < eps) or \
                (stop_criterion == 'value' and abs(next_value - cur_value) < eps) or \
                (stop_criterion == 'delta' and np.linalg.norm(cur_delta) < eps):
            return trace
        cur_arg = next_arg
        cur_value = next_value


if __name__ == '__main__':
    # task_1()
    task_2()
