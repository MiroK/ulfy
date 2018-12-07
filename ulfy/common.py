from ufl.core.terminal import Terminal
from sympy import Expr, Matrix, symbols
from numpy import number


def is_number(f):
    '''Numbers'''
    return isinstance(f, (int, float, number))


def is_terminal(f):
    '''UFL terminals'''
    return isinstance(f, Terminal)


def is_matrix(f):
    '''Matrices including (1, 1)'''
    # return isinstance(f, Matrix) and (f.shape == (1, 1) or 1 not in f.shape)
    return isinstance(f, Matrix) and 1 not in f.shape


def is_vector(f):
    '''Row or column vectors'''
    return isinstance(f, Matrix) and not is_matrix(f)


def is_scalar(f):
    '''Scalar are numbers, non-number expressions are 0 dim arrays'''
    return is_number(f) or isinstance(f, Expr) or f.shape == ()


def str_to_num(string):
    '''Convert (list of) string to (list of) numbers'''
    # WHY: To make expression work we return possible constants as strings
    # then when we are certain that we have a constant expression we convert
    if isinstance(string, str):
        return float(string)
    return tuple(map(str_to_num, string))


DEFAULT_NAMES = symbols('x y z')
