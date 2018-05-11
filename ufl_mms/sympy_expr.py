from sympy.printing import ccode
import dolfin as df
import sympy as sp

from common import is_scalar, is_number, is_vector, is_matrix, is_terminal, DEFAULT_NAMES
from ufl_sympy import ufl_to_sympy, DEFAULT_RULES


def expr_body(expr, coordnames=DEFAULT_NAMES, **kwargs):
    '''Generate a/list of string/s that is the Cpp code for the expression'''
    if is_number(expr):
        return expr_body(sp.S(expr), **kwargs)

    if isinstance(expr, sp.Expr) and is_scalar(expr):
        # Defined in terms of some coordinates
        xyz = set(coordnames)
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used)
        # Substitute for x[0], x[1], ...
        expr = expr.subs({x: sp.Symbol('x[%d]' % i) for i, x in enumerate(coordnames)},
                         simultaneous=True)
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # Default to zero
        kwargs.update(dict((str(p), kwargs.get(str(p), 0)) for p in params))
        # Convert
        return expr, kwargs
    
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
    ans = ()
    for e in expr:
        f, kwargs_ = expr_body(e, **kwargs_)
        ans = ans + (f, )
    return ans, kwargs_


def check_substitutions(subs):
    '''Subs: UFL terminals -> sympy expressions of right type'''
    if not all(map(is_terminal, subs.keys())):
        return False

    check_scalar = lambda k, v: k.ufl_shape == () and (is_scalar(v) or is_number(v))

    check_vector = lambda k, v: ((len(k.ufl_shape) == 1 and is_vector(v))
                                 and
                                 (k.ufl_shape[0] in (v.rows, v.cols)))

    check_matrix = lambda k, v: len(k.ufl_shape) == 2 and k.ufl_shape == (v.rows, v.cols)

    check = lambda p: check_scalar(*p) or check_vector(*p) or check_matrix(*p)

    return all(map(check, subs.items()))


def Expression(body, **kwargs):
    '''Construct dolfin.Expression from sympy/ufl expressions'''
    # Generate body and ask again
    if isinstance(body, (sp.Expr, sp.Matrix)):
        body, kwargs = expr_body(body, **kwargs)
        return Expression(body, **kwargs)

    # Translare UFL and ask again
    if hasattr(body, 'ufl_shape'):
        subs = kwargs.pop('subs')
        # Make sure that the UFL terminal are mapped to sensible sympy things
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
        for f in subs.keys():  # Possible expressions
            params = getattr(f, 'user_parameters', {})
            
            if not isinstance(f, df.Expression): continue
            
            for p in params:
                # In the end all the params will be put to one dict so
                # they better agree on their values (think t as time)
                if p in user_parameters:
                    assert user_parameters[p] == getattr(v, p)  # V's current
                else:
                    user_parameters[p] = getattr(f, p)
        # However explicit parameters take precedence
        for p, v in user_parameters.items():
            if p not in kwargs: kwargs[p] = v
        # Build it
        return Expression(body, **kwargs)
    
    # We have strings, lists and call dolfin
    return df.Expression(body, **kwargs)
