from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.conditional import EQ, NE, GT, LT, GE, LE, MaxValue, MinValue
from functools import reduce
import ufl, dolfin, sympy, operator
from random import sample
import numpy as np

from .common import *


def random_symbol(used, size=4):
    '''Name of size avoiding used'''
    alphabet = list(map(chr, range(65, 91)))
    alphabet.extend(list(map(lambda s: s.lower(), alphabet)))
    
    while True:
        symbol = sympy.Symbol(''.join(sample(alphabet, size)))
        if symbol not in used:
            return symbol

        
def make_rule(rule):
    '''
    Returns a function which uses `rule` to create sympy expression from 
    UFL expr operands translated to sympy expressions.
    '''
    def apply_rule(expr, subs, rules):
        # Reduce to sympy
        operands = tuple(ufl_to_sympy(o, subs, rules) for o in expr.ufl_operands)
        # Sympy action
        return rule(*operands)
    return apply_rule


def terminal_rule(expr, subs, rules, coordnames=DEFAULT_NAMES):
    '''Translate UFL terminals'''
    # Compute
    if isinstance(expr, (ufl.algebra.ScalarValue, ufl.algebra.IntValue)):
        return expr.value()

    if isinstance(expr, ufl.constantvalue.Zero):
        if expr.ufl_shape == ():
            return 0
        else:
            return sympy.Matrix(np.zeros(expr.ufl_shape))

    if isinstance(expr, dolfin.Constant):
        if expr.ufl_shape == ():
            if expr in subs:
                # Make constant scalars into symbols which will take the
                # value from the dictionary. This way Expression('A', A=2)
                # is made so we don't recompile
                # used = reduce(operator.and_,
                #               (v.free_symbols if hasattr(v, 'free_symbols') else set() for v in subs.values()))
                # symbol = random_symbol(used, size=3)
                # # Make nute of the substitution
                # print(symbol, expr(0))
                return subs[expr]
            else:
                return expr(0)
        else:
            if not expr in subs:
                return sympy.Matrix(np.array(expr.values()).reshape(expr.ufl_shape))
            else:
                base = subs[expr]
                if len(expr.ufl_shape) == 1:
                    n, = expr.ufl_shape
                    syms = [sympy.Symbol(f'{base}_{i}') for i in range(n)]
                else:
                    nrows, ncols = expr.ufl_shape
                    syms = [sympy.Symbol(f'{base}_{i}{j}') for i in range(nrows) for j in range(ncols)]

                return sympy.Matrix(np.array(syms).reshape(expr.ufl_shape))
            
            
    if isinstance(expr, ufl.constantvalue.Identity):
        return sympy.eye(expr.ufl_shape[0])

    if isinstance(expr, ufl.geometry.SpatialCoordinate):
        return sympy.Matrix(coordnames[:expr.ufl_shape[0]]) # A column vector

    # Look it up
    return subs[expr]


def grad_rule(expr, subs, rules, coordnames=DEFAULT_NAMES):
    '''Translate gradient expression'''
    f, = expr.ufl_operands
    gdim = expr.ufl_shape[-1]
    # Reduce to sympy
    f = ufl_to_sympy(f, subs, rules)
    # Consider gdim coords when differentiating
    scalar_grad = lambda f, x=coordnames[:gdim]: [f.diff(xi) for xi in x]

    if is_scalar(f):
        return sympy.Matrix(scalar_grad(f))
    return sympy.Matrix(list(map(scalar_grad, f)))


def nabla_grad_rule(expr, subs, rules, coordnames=DEFAULT_NAMES):
    '''Translate nabla gradient expression'''
    f, = expr.ufl_operands
    gdim = expr.ufl_shape[-1]
    # Reduce to sympy
    f = ufl_to_sympy(f, subs, rules)
    # Consider gdim coords when differentiating
    scalar_grad = lambda f, x=coordnames[:gdim]: [f.diff(xi) for xi in x]

    if is_scalar(f):
        return sympy.Matrix(scalar_grad(f))
    return sympy.Matrix(list(map(scalar_grad, f))).T


def div_rule(expr, subs, rules, coordnames=DEFAULT_NAMES):
    '''Translate div expression'''
    f, = expr.ufl_operands
    # Reduce to sympy
    f = ufl_to_sympy(f, subs, rules)
    
    vector_div = lambda f, x=coordnames: sum(fi.diff(xi) for fi, xi in zip(f, x))

    if is_vector(f):
        return vector_div(f)
    # Row wise d s_{ij}/d x_j
    return sympy.Matrix(list(map(vector_div, [f[i, :] for i in range(f.rows)])))


def nabla_div_rule(expr, subs, rules, coordnames=DEFAULT_NAMES):
    '''Translate nabla div expression'''
    f, = expr.ufl_operands
    # Reduce to sympy
    f = ufl_to_sympy(f, subs, rules)
    
    vector_div = lambda f, x=coordnames: sum(fi.diff(xi) for fi, xi in zip(f, x))

    if is_vector(f):
        return vector_div(f)
    # Row wise d s_{ij}/d x_i
    return sympy.Matrix(list(map(vector_div, [f[:, i] for i in range(f.rows)])))


def curl_rule(expr, subs, rules, coordnames=DEFAULT_NAMES):
    '''Translate curl expression'''
    f, = expr.ufl_operands
    shape = f.ufl_shape
    # Reduce to sympy
    f = ufl_to_sympy(f, subs, rules)
    # TODO: how are the definitions for manifolds?
    x = coordnames
    # Curl of scalar (should be 2d) is a vector
    if shape == ():
        return sympy.Matrix([f.diff(x[1]), -f.diff(x[0])])
    # Curl of 2d vector is div(R f) where R is a rotation matrix
    elif shape == (2, ):
        return f[1].diff(x[0]) - f[0].diff(x[1])
    # Usral 3d vec curl
    elif shape == (3, ):
        return sympy.Matrix([f[2].diff(x[1]) - f[1].diff(x[2]),
                             f[0].diff(x[2]) - f[2].diff(x[0]),
                             f[1].diff(x[0]) - f[0].diff(x[1])])
    else:
        raise ValueError("%s %s %r" % (f, type(f), shape))

    
def indexed_rule(expr, subs, rules):
    '''Index node is Constant((2, 3))[0]'''
    f, indices = expr.ufl_operands
    # This rule is correct when all indices are fixed otherwise we return
    # something that e.g. IndexSum can use
    f, indices = expr.ufl_operands
    # Compute indices
    shape = f.ufl_shape
    indices = tuple(slice(l) if isinstance(index, ufl.indexed.Index) else int(index)
                    for l, index in zip(shape, indices))
    # Get what to index
    f = ufl_to_sympy(f, subs, rules)
    # Slice
    if len(indices) == 1:
        index = indices[0]
        return f[index] if isinstance(index, int) else sympy.Matrix(f[index])
    return sympy.Matrix(f[indices])


def component_tensor_rule(expr, subs, rules):
    '''ComponentTensor is Identity(3)[:, 2]'''
    indexed, indices = expr.ufl_operands
    # FIXME: This is not general enough
    indexed, _ = expr.ufl_operands
    return ufl_to_sympy(indexed, subs, rules)


def index_sum_rule(expr, subs, rules):
    '''Index sum compiles into a function'''
    body, summation_indices = expr.ufl_operands
    assert isinstance(body, ufl.algebra.Product)
    # Eval arguments of product
    a, b = body.ufl_operands
    # These are both indexed. I assume that these are A*B aor A*v and
    # then in either case dot is the way to go
    a = ufl_to_sympy(a, subs, rules)
    b = ufl_to_sympy(b, subs, rules)
    # This is really ugly
    if is_vector(a): return b*a

    return a*b

# Some of the tensor algebra rules will be computed using numpy.

def to_numpy(a):
    '''Sympy Matrix to numpy conversion'''
    A = np.array(a.tolist())
    if is_matrix(a):
        return A
    # Sympy has vectors as (n, 1) or (1, n) hence flatten
    return A.flatten().reshape((np.prod(a.shape), ))


def with_numpy(op, *args):
    '''Apply op to args converted to array'''
    args = list(map(to_numpy, args))
    ans = op(*args)

    if not is_scalar(ans):
        return sympy.Matrix(ans)
    else:
        if is_number(ans) or isinstance(ans, sympy.Expr):
            return ans
        # Eg. tensordot results in array is () shape
        else:
            return ans.tolist()

# These are too long to be lambdas
        
def _inner(a, b):
    '''Translate inner node'''
    # Conract matrices
    if is_matrix(a) and is_matrix(b):
        return with_numpy(np.tensordot, a, b)
    # (num|vec) * (num|vec)
    return with_numpy(np.inner, a, b)

    
def _list_tensor(*comps):
    '''Comes from as_vector or as_matrix'''
    if is_scalar(comps[0]):
        return sympy.Matrix(comps)
    # Sympy goes to col vectors so make them row and then stack up
    return sympy.Matrix([c.T for c in comps])


# Mapping of nodes
# I start with exception which are not done via make_rule
DEFAULT_RULES = {
    # Calculus
    ufl.differentiation.Grad: grad_rule,
    ufl.differentiation.Div: div_rule,
    ufl.differentiation.Curl: curl_rule,
    ufl.differentiation.NablaGrad: nabla_grad_rule,
    ufl.differentiation.NablaDiv: nabla_div_rule,
    # Delegate job
    # TODO here I disregard label
    ufl.differentiation.Variable: lambda e, s, r: ufl_to_sympy(e.ufl_operands[0], s, r),
    # Diff and try again
    ufl.differentiation.VariableDerivative: lambda e, s, r: ufl_to_sympy(apply_derivatives(e), s, r),
    # Indexing
    ufl.indexed.Indexed: indexed_rule,
    ufl.tensors.ComponentTensor: component_tensor_rule,
    ufl.indexsum.IndexSum: index_sum_rule,
}
# And now the rest
DEFAULT_RULES.update(
    dict((node, make_rule(rule)) for (node, rule) in
         (
             # Algebra
             (ufl.algebra.Sum, lambda a, b: a+b),
             (ufl.algebra.Abs, lambda a: abs(a)),
             (ufl.algebra.Division, lambda a, b: a/b),  
             (ufl.algebra.Product, lambda a, b: a*b),
             (ufl.algebra.Power, lambda a, b: a**b),
             (ufl.algebra.Conj, lambda a: a),
             # Tensor algebra
             (ufl.tensoralgebra.Determinant, lambda a: a.det()),
             (ufl.tensoralgebra.Inverse, lambda a: a.inv()),
             (ufl.tensoralgebra.Transposed, lambda a: a.T),
             (ufl.tensoralgebra.Trace, lambda a: a.trace()),
             (ufl.tensoralgebra.Sym, lambda a: (a + a.T)/2),
             (ufl.tensoralgebra.Skew, lambda a: (a - a.T)/2),
             (ufl.tensoralgebra.Deviatoric, lambda a: a - a.trace()*sympy.eye(a.rows)/a.rows),
             (ufl.tensoralgebra.Cofactor, lambda a: a.det()*(a.inv().T)),
             (ufl.tensoralgebra.Cross, lambda a, b: a.cross(b)),
             (ufl.tensoralgebra.Outer, lambda a, b: with_numpy(np.outer, a, b)),
             (ufl.tensoralgebra.Dot, lambda a, b: with_numpy(np.dot, a, b)),
             (ufl.tensoralgebra.Inner, _inner),
             # Math functions of one argument
             (ufl.mathfunctions.Sin, sympy.sin),
             (ufl.mathfunctions.Cos, sympy.cos),
             (ufl.mathfunctions.Sqrt, sympy.sqrt),
             (ufl.mathfunctions.Exp, sympy.exp),
             (ufl.mathfunctions.Ln, sympy.log),
             (ufl.mathfunctions.Tan, sympy.tan),
             (ufl.mathfunctions.Sinh, sympy.sinh),
             (ufl.mathfunctions.Cosh, sympy.cosh),
             (ufl.mathfunctions.Tanh, sympy.tanh),
             (ufl.mathfunctions.Asin, sympy.asin),
             (ufl.mathfunctions.Acos, sympy.acos),
             (ufl.mathfunctions.Atan, sympy.atan),
             (ufl.mathfunctions.Atan2, sympy.atan2),
             (ufl.mathfunctions.Erf, sympy.erf),
             # Math functions of two arguments
             (ufl.mathfunctions.BesselI, sympy.functions.special.bessel.besseli),
             (ufl.mathfunctions.BesselY, sympy.functions.special.bessel.bessely),
             (ufl.mathfunctions.BesselJ, sympy.functions.special.bessel.besselj),
             (ufl.mathfunctions.BesselK, sympy.functions.special.bessel.besselk),
             # Boolean
             (EQ, sympy.Eq), (NE, sympy.Ne), (GT, sympy.Gt), (LT, sympy.Lt), (GE, sympy.Ge), (LE, sympy.Le),
             # Conditionals
             (ufl.operators.AndCondition, sympy.And),
             (ufl.operators.OrCondition, sympy.Or),
             (ufl.operators.NotCondition, sympy.Not),
             (ufl.operators.Conditional, lambda c, t, f: sympy.Piecewise((t, c), (f, True))),
             #
             (MaxValue, sympy.Max),
             (MinValue, sympy.Min),
             # Indexing    
             (ufl.tensors.ListTensor, _list_tensor)
         )
    )
)


def ufl_to_sympy(expr, subs, rules=DEFAULT_RULES):
    '''
    Translate UFL expression to sympy expression according to rules and 
    using expressions in subs to replace terminals
    '''
    # Primitives
    # UFL terminals
    if is_terminal(expr):
        return terminal_rule(expr, subs, rules)
    # Uncaught numbers; identity - t
    if is_number(expr):
        return ufl_to_sympy(Constant(expr), subs, rules)
    # Translate if it wasn't done
    if expr not in subs:
        subs[expr] = rules[type(expr)](expr, subs, rules)
    # Lookup
    return subs[expr]
