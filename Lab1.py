import random
from math import sqrt
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
import progressbar


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
        if iterations > 100:
            break
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
        if iterations > 100:
            break
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

        if iterations > 100:
            break
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

        if iterations > 100:
            break

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


def seadle(arg):
    x, y = arg
    return x ** 6 - x ** 3 + y ** 2


def grad_seadle(arg):
    x, y = arg
    return 6 * x ** 5 - 3 * x ** 2, 2 * y


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

        if iterations > 2000:
            break
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
    start = np.random.normal(loc=0., scale=2., size=2)

    steps = [
        linear_step(dichotomy),
        linear_step(lambda a, b, f, e: straight(f, golden_section, e)),
        linear_step(golden_section),
        linear_step(fibonacci),
        constant_step(0.001),
        armiho()
    ]

    for f, df in [(circle, grad_circle), (seadle, grad_seadle)]:
        for step in steps:
            m = gradient_descent(f, df, start, step, 0.0001)
            trace = m["trace"]
            m.pop("trace")
            fig = plt.figure(figsize=(6, 5))
            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_axes([left, bottom, width, height])
            x_val = np.arange(-5, 5, 0.01)
            y_val = np.arange(-5, 5, 0.01)
            X, Y = np.meshgrid(x_val, y_val)

            arg = [f([x, y]) for x, y in zip(X, Y)]

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


def create_matrix(c, n):
    r = sqrt(c)
    A = np.random.randn(n, n)
    u, s, v = np.linalg.svd(A)
    h, l = np.max(s), np.min(s)  # highest and lowest eigenvalues (h / l = current cond number)

    # linear stretch: f(x) = a * x + b, f(h) = h, f(l) = h/r, cond number = h / (h/r) = r
    def f(x):
        return h * (1 - ((r - 1) / r) / (h - l) * (h - x))

    new_s = f(s)
    new_A = (u * new_s) @ v.T  # make inverse transformation (here cond number is sqrt(k))
    new_A = new_A @ new_A.T  # make matrix symmetric and positive semi-definite (cond number is just k)
    return new_A


def matrix_fn(c, n):
    A = create_matrix(c, n)
    b = np.random.randn(len(A))
    f = lambda x: x.dot(A).dot(x) - b.dot(x)
    f_grad = lambda x: (A + A.T).dot(x) - b
    return A, f, f_grad


def estimate(c, n, step_chooser, eps=0.0001, n_checks=20):
    avg_iters = 0
    name = ""
    for _ in range(n_checks):
        A, f, df = matrix_fn(c, n)
        init_x = np.random.randn(len(A))

        m = gradient_descent(f, df, init_x, step_chooser, eps)
        name = m["name"]
        avg_iters += len(m["trace"])
    return avg_iters / n_checks, name


def task_3():
    steps = [
        linear_step(dichotomy),
        linear_step(lambda a, b, f, e: straight(f, golden_section, e)),
        linear_step(golden_section),
        linear_step(fibonacci),
        # constant_step(0.001),
        armiho()
    ]
    i = 0
    with progressbar.ProgressBar(max_value=5 * len(steps) * 25) as bar:
        for n in [3, 10, 15, 20]:
            fig, ax = plt.subplots()
            ax.set_title("Number of iterations, n=%d" % n)
            for step in steps:
                y = []
                x = []
                name = ""
                for k in np.linspace(1, 1000, 25):
                    i += 1
                    bar.update(i)
                    r, name = estimate(k, n, step)
                    y.append(r)
                    x.append(k)
                ax.plot(x, y, label=name)
                ax.legend()
            plt.show()


def read_dataset(path):
    data = pd.read_csv(path)
    X = data.iloc[:,3:-1].values
    y = data.iloc[:, -1].apply(lambda c: 1 if c == 'P' else -1).values
    return X, y


def calc_f_score(X, y, alpha, solver):
    n_splits = 5
    cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)
    mean_f_score = 0.0
    for train_indexes, test_indexes in cv.split(X):
        X_train = X[train_indexes]
        X_test = X[test_indexes]
        y_train = y[train_indexes]
        y_test = y[test_indexes]

        classifier = LogReg(alpha, solver)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test != 1))
        tn = np.sum((y_pred != 1) & (y_test != 1))
        fn = np.sum((y_pred != 1) & (y_test == 1))

        if tp != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * precision * recall / (precision + recall)
            mean_f_score += f_score
    return mean_f_score / n_splits


def get_best_param(X, y, solver):
    best_alpha = None
    max_f_score = -1
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.]:
        cur_f_score = calc_f_score(X, y, alpha, solver)
        print('alpha =', alpha, 'f-score =', cur_f_score)
        if cur_f_score > max_f_score:
            max_f_score = cur_f_score
            best_alpha = alpha
    return best_alpha, max_f_score


def process_with_solver(X, y, solver, step_x, step_y):
    best_alpha, max_f_score = get_best_param(X, y, solver)
    print('Best params:', best_alpha, max_f_score)
    best_classifier = LogReg(best_alpha, solver)
    start_time = datetime.now()
    errors, steps = best_classifier.fit(X, y)
    end_time = datetime.now()
    timedelta = end_time - start_time
    if solver == 'newton':
        print('steps =', steps)
    else:
        print('steps =', steps)
    print('time =', timedelta.microseconds / 1000.0, 'ms')


def task_4a():
    X, y = read_dataset('vowel.csv')
    process_with_solver(X, y, 'gradient', 0.1, 0.01)


def task_4b():
    X, y = read_dataset('vowel.csv')
    process_with_solver(X, y, 'newton', 0.1, 0.01)


# if __name__ == '__main__':
    # task_1()
    # task_2()
    # task_4()

