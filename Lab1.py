import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


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


def circle(x, y):
    return (y - 1) ** 2 + (x + 2) ** 2 / 4 + x + y + 1


def grad_circle(x, y):
    return (x + 4) / 2, 2 * y - 1


def trace_grad_min(f, df, search, eps):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)

    x_prev = None
    y_prev = None

    trace = [x, y]

    iterations = 0
    calc = 0
    name = None

    while x_prev is None or sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2) >= eps:
        print("x: {}, y: {}".format(x, y))
        dx, dy = df(x, y)
        iterations += 1

        def best(s):
            return f(x + dx * s, y + dy * s)

        m = search(-1000, 1000, best, eps)
        step = m["answer"]
        name = m["name"]
        calc += m["calc"]

        x_prev = x
        y_prev = y

        x = x + dx * step
        y = y + dy * step

        trace.append((x, y))
    return {"trace": trace, "calc": calc, "iterations": iterations, "name": name}


def task_2():
    for method in [dichotomy, lambda a, b, f, e: straight(f, golden_section, e), golden_section, fibonacci]:
        m = trace_grad_min(circle, grad_circle, method, 0.0001)
        trace = m["trace"]
        m.pop("trace")
        fig = plt.figure(figsize=(6, 5))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        x_vals = np.arange(-10, 10, 0.1)
        y_vals = np.arange(-10, 10, 0.1)
        X, Y = np.meshgrid(x_vals, y_vals)

        cp = ax.contour(X, Y, circle(X, Y))
        ax.clabel(cp, inline=True, fontsize=10)
        ax.set_title(m["name"])
        plt.show()
        print(m)


if __name__ == '__main__':
    # task_1()
    task_2()
