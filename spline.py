from functions import *
import numpy as np


class MinimalSpline:
    def __init__(self, gener_fun, grid):
        self.x = sp.symbols('t')
        self._grid = grid
        if gener_fun == "const":
            self.generating_function = const()
        elif gener_fun == "tan":
            self.generating_function = tan()
        elif gener_fun == "sqrt":
            self.generating_function = sqrt()

    @staticmethod
    def left_biorthogonal_functionals(spline, x_sample, y_sample, indices):
        coeffs = [y_sample[indices[i]] / spline.generating_function.subs(spline.x, x_sample[indices[i]]) for i in
                  range(0, len(indices) - 1)]
        return coeffs

    @staticmethod
    def central_biorthogonal_functionals(spline, x_sample, y_sample, indices):
        step = int((len(y_sample) - 1) / (2 * (len(indices) - 1)))
        coeffs = [y_sample[indices[i] + step] / spline.generating_function.subs(spline.x, x_sample[indices[i] + step])
                  for i in range(0, len(indices) - 1)]
        return coeffs

    @staticmethod
    def balanced_coeffs(spline, x_sample, y_sample, indices):
        gen = [[spline.generating_function.subs(spline.x, x_sample[j]) for j in
                         range(indices[i - 1], indices[i] + 1)] for i in range(1, len(indices))]
        coeffs = [(max(y_sample[indices[i - 1]: indices[i] + 1]) + min(y_sample[indices[i - 1]: indices[i] + 1])) / (
                #     max([spline.generating_function.subs(spline.x, x_sample[j]) for j in
                #          range(indices[i - 1], indices[i] + 1)]) + min(
                # [spline.generating_function.subs(spline.x, x_sample[j]) for j in
                #  range(indices[i - 1], indices[i] + 1)])) for i in range(1, len(indices))]
                max(gen[i - 1]) + min(gen[i - 1]))
                  for i in range(1, len(indices))]
        return coeffs

    def __call__(self, x_sample, y_sample, indices, coeff):
        _coeffs = coeff(self, x_sample, y_sample, indices)
        knots = x_sample[indices]
        pieces = [(self._grid >= knot) for knot in knots[:-1]]
        approximation = np.piecewise(self._grid, pieces, [sp.lambdify(self.x, coeff * self.generating_function) for coeff in _coeffs])
        return approximation
