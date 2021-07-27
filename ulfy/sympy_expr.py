from sympy.printing import ccode
import dolfin as df
import sympy as sp

from .common import (is_scalar, is_number, is_vector, is_matrix, is_terminal,
                    str_to_num, DEFAULT_NAMES)
from .ufl_sympy import ufl_to_sympy, DEFAULT_RULES
from ufl.differentiation import Variable


def expr_body(expr, coordnames=DEFAULT_NAMES, **kwargs):
    '''Generate a/list of string/s that is the Cpp code for the expression'''
    if is_number(expr):
        return expr_body(sp.S(expr), **kwargs)

    if isinstance(expr, sp.Expr) and is_scalar(expr):
        # Defined in terms of some coordinates
        xyz = set(coordnames)
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        
        # Recognize the constant
        if not expr.free_symbols:
            # Flag that we can be constant
            return str(expr), kwargs, True
        
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used)
        # Substitute for x[0], x[1], ...
        expr = expr.subs({x: sp.Symbol('x[%d]' % i) for i, x in enumerate(coordnames)},
                         simultaneous=True)
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # log needs to be replaced by std::log to avoid confusion with the log() function in Dolfin
        expr = expr.replace("log", "std::log")
        # Default to zero
        kwargs.update(dict((str(p), kwargs.get(str(p), 0)) for p in params))
        # Convert
        return expr, kwargs, False
    
    # Tensors that sympy can represent as lists
    # (1, n) to (n, 1) to list of n
    if is_vector(expr):
        expr = sum(expr.tolist(), [])
    elif is_matrix(expr):
        expr = expr.tolist()
        
    # Other lists
    # FIXME: Can this be implemented without returning kwargs, i.e. the
    # scalar place would modify it's arguments. For now I don't see how
    # https://stackoverflow.com/questions/45883655/is-it-always-safe-to-modify-the-kwargs-dictionary
    kwargs_ = kwargs
    is_constant_expr = True
    ans = ()
    for e in expr:
        f, kwargs_, is_constant = expr_body(e, **kwargs_)
        is_constant_expr = is_constant_expr and is_constant
        ans = ans + (f, )
    return ans, kwargs_, is_constant_expr


def check_substitutions(subs):
    '''Subs: UFL terminals/variable -> sympy expressions of right type'''
    if not all(is_terminal(k) or isinstance(k, Variable) for k in list(subs.keys())):
        return False

    # If the form is defined in terms of vars as well as terminals we inject
    # unwrapped variables
    subs.update({k.ufl_operands[0]: v for k, v in list(subs.items()) if isinstance(k, Variable)})

    check_scalar = lambda k, v: k.ufl_shape == () and (is_scalar(v) or is_number(v))

    check_vector = lambda k, v: ((len(k.ufl_shape) == 1 and is_vector(v))
                                 and
                                 (k.ufl_shape[0] in (v.rows, v.cols)))

    check_matrix = lambda k, v: len(k.ufl_shape) == 2 and k.ufl_shape == (v.rows, v.cols)

    check = lambda p: check_scalar(*p) or check_vector(*p) or check_matrix(*p)

    return all(map(check, list(subs.items())))


def Expression(body, **kwargs):
    '''Construct dolfin.Expression or Constant from sympy/ufl expressions'''
    # Generate body and ask again
    if isinstance(body, (sp.Expr, sp.Matrix, sp.ImmutableMatrix)):
        body, kwargs, is_constant_expr = expr_body(body, **kwargs)
        return df.Constant(str_to_num(body)) if is_constant_expr else Expression(body, **kwargs)

    # Translare UFL and ask againx
    if hasattr(body, 'ufl_shape'):
        subs = kwargs.pop('subs').copy()
        # Make sure that we start with dictionary of terminals mapped to
        # sensible values. Note that the subs dict will grow during translation
        assert check_substitutions(subs)
        # Collect arguments for UFL conversion
        if 'rules' in kwargs:
            rules = kwargs.pop('rules')
        else:
            rules = DEFAULT_RULES
        body = ufl_to_sympy(body, subs, rules)

        # If the subs are expressions with parameters then it is convenient
        # to build the result with values of their parameters set to the
        # current value (otherwise all params are set to 0)
        user_parameters = {}
        for f in list(subs.keys()):  # Possible expressions
            if not hasattr(f, 'user_parameters'):
                params = {}
            else:
                params = f.user_parameters.__dict__['_params']

            if not isinstance(f, df.Expression): continue
            
            for p in params:
                # In the end all the params will be put to one dict so
                # they better agree on their values (think t as time)
                if p in user_parameters:
                    assert user_parameters[p] == getattr(v, p)  # V's current
                else:
                    user_parameters[p] = getattr(f, p)
        # However explicit parameters take precedence
        for p, v in list(user_parameters.items()):
            if p not in kwargs: kwargs[p] = v
        # Build it
        return Expression(body, **kwargs)
    
    # We have strings, lists and call dolfin
    return df.Expression(body, **kwargs)
