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
    return {"answer": (a + b) / 2, "iterations": iterations, "calc": calc}


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
    return {"answer": (a + b) / 2, "calc": calc, "iterations": iterations}


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
    x1 = a + number_fibonacci(n - k + 1) / number_fibonacci(n + 2) * (b - a)
    x2 = a + number_fibonacci(n - k + 2) / number_fibonacci(n + 2) * (b - a)
    fx1 = g(x1)
    fx2 = g(x2)
    while k <= n and b - a >= eps:
        if fx1 < fx2:
            b = x2
            x2 = x1
            x1 = a + number_fibonacci(n - k + 1) / number_fibonacci(n + 2) * (b - a)
            fx2 = fx1
            fx1 = g(x1)
        if fx2 < fx1:
            a = x1
            x1 = x2
            x2 = a + number_fibonacci(n - k + 2) / number_fibonacci(n + 2) * (b - a)
            fx1 = fx2
            fx2 = g(x2)
        k = k + 1
    return {"answer": (a + b) / 2, "calc": calc, "iterations": iterations}
