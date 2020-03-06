from math import sqrt


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
        if fx2 < fx1:
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
        elif fx2 < fx1:
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
        lambda e: dichotomy(a, b, f, e),
        lambda e: fibonacci(a, b, f, e),
        lambda e: golden_section(a, b, f, e),
        lambda e: straight(f, golden_section, e)
    ]

    for method in methods:
        for e in eps:
            print(method(e))


if __name__ == '__main__':
    task_1()
