import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from spline import *
import numpy as np


def var(y):
    diffs = np.diff(y)
    signs = [math.copysign(1, diff) for diff in diffs]
    extremums = [0]
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            extremums.append(i)
    extremums.append(len(y) - 1)
    _var = sum([abs(y[extremums[i]] - y[extremums[i - 1]]) for i in range(1, len(extremums))])
    return _var


def balanced_grid(x, y, n):
    all_var = var(y) / n
    indices = [0]
    potential_knot = 0
    for i in range(indices[-1] + 1, len(x) + 1):
        _var = var(y[indices[-1]:i])
        if abs(_var - all_var) < n / len(x):
            if potential_knot != 0:
                if abs(_var - all_var) <= abs(cur_best - all_var):
                    cur_best = _var
                    potential_knot = i - 1
                else:
                    indices.append(potential_knot)
                    potential_knot = 0
                    cur_best = 0
            elif potential_knot == 0:
                potential_knot = i - 1
                cur_best = _var
        else:
            if potential_knot != 0:
                indices.append(potential_knot)
                potential_knot = 0
                cur_best = 0
    if indices[-1] == len(x) - 2:
        del indices[-1]
    if indices[-1] != len(x) - 1:
        indices.append(len(x) - 1)
    return indices


def uniform_grid(x, n):
    _n = (len(x)) // n
    indices = np.arange(len(x))[0::_n]
    return indices


def error(data, approximation, knots):
    size = len(approximation) - 1
    indices = [round(i * (size / (10 * knots))) for i in range(0, 10 * knots + 1)]
    diff = abs((approximation[indices] - data[indices]))
    return max(diff)


def save_table(x, y, gen_funcs, knots, coeff, grid, file_name):
    headers = ["$\phi(t)$"]
    table = []
    for i in knots:
        headers.append(f'n={i}')
    for gen in gen_funcs:
        spline = MinimalSpline(gen, x)
        table.append([sp.latex(spline.generating_function)])
        for i in knots:
            if grid == "balanced":
                approx_grid = balanced_grid(x, y, i)
            elif grid == "uniform":
                approx_grid = uniform_grid(x, i)
            approximation = spline(x, y, approx_grid, coeff)
            err = error(approximation, y, i)
            table[-1].append(err)
    with open("Results/" + file_name, "w") as file:
        file.write(tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".6f"))


def draw_plots(x, y, func_name, gen_funcs, gen_names, knots, coeff, grid):
    plt.plot(x, y, label=f'{func_name}')
    for i in range(0, len(gen_funcs)):
        spline = MinimalSpline(gen_funcs[i], x)
        if grid == "balanced":
            approx_grid = balanced_grid(x, y, knots)
        elif grid == "uniform":
            approx_grid = uniform_grid(x, knots)
        approximation = spline(x, y, approx_grid, coeff)
        plt.plot(x, approximation, label=f'{gen_names[i]}')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    x = np.linspace(0, 1, 10001)
    gen_funcs = ["const", "tan", "sqrt"]
    knots = [10, 50, 100]

    y = np.exp(x)
    save_table(x, y, gen_funcs, knots, MinimalSpline.left_biorthogonal_functionals, grid="uniform",
               file_name="exp_left_biorth_functionals.txt")
    y = np.sinh(x)
    save_table(x, y, gen_funcs, knots, MinimalSpline.left_biorthogonal_functionals, grid="uniform",
               file_name="sinh_left_biorth_functionals.txt")

    y = np.exp(x)
    save_table(x, y, gen_funcs, knots, MinimalSpline.central_biorthogonal_functionals, grid="uniform",
               file_name="exp_central_biorth_functionals.txt")
    y = np.sinh(x)
    save_table(x, y, gen_funcs, knots, MinimalSpline.central_biorthogonal_functionals, grid="uniform",
               file_name="sinh_central_biorth_functionals.txt")

    y = np.exp(x)
    save_table(x, y, gen_funcs, knots, MinimalSpline.balanced_coeffs, grid="balanced",
               file_name="exp_on_balanced_const_knots.txt")
    y = np.sinh(x)
    save_table(x, y, gen_funcs, knots, MinimalSpline.balanced_coeffs, grid="balanced",
               file_name="sinh_on_balanced_const_knots.txt")
