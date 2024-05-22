import sympy as sp


def const():
    func = 1
    return sp.sympify(func)


def tan():
    x = sp.symbols("t")
    func = sp.tan(x) + sp.pi
    return sp.sympify(func)


def sqrt():
    x = sp.symbols("t")
    func = sp.sqrt(2 + x)
    return sp.sympify(func)