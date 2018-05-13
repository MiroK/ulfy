# ulfy

This package translates UFL expressions to SymPy expressions which can then be compiled to instances of DOLFIN Expressions. A typical use case is when one wants to prepare a manufactured solution for their FEniCS solver. 

````python
from dolfin import *
from ulfy import Expression
import sympy as sp

x, y, z, t = sp.symbols('x y z t')
f = 3*x*y + x**2 + 2*y**2 + t**3

mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, 'CG', 2)
v = Function(V)

e = v - 3*v**2 + v.dx(0)  # UFL
e = Expression(e, subs={v: f}, degree=4)  # to DOLFIN
````
## Rationale
Assuming you started with SymPy you will most likely write some small ad-hoc vector calculus module to deal with `grad`, `div`, etc to get the desired right hand side. If you are like me you will end up writing such module for each solver and that is no fun. Alternatively, you can build the right hand side as an UFL expression, however, to use it with `errornorm` the expression needs to be projected (as UFL expressions cannot be interpolated) and this involves a solve of a linear system. Your third (and _best_ ) option is to use `ulfy` and translate UFL expression, e.g. `-div(grad(u))` (for the right hand side of Poisson problem) to the DOLFIN expression. Note that expressions such as `u+u` can be handled by DOLFIN; `Expression('u+u', u=u, degree=1)`.

## What works
At the moment `ulfy` supports most commonly used nodes in UFL, (see the [here](https://github.com/MiroK/ulfy/blob/master/tests/test_ufl_mms.py) for examples). Nodes for `ComponentTensor` and `IndexSum` are only partially supported. More precisely, let
`A=Costant(((1, 2), (3, 4)))`. Then `as_matrix(A[i, j], [j, i])` will not work but `A*A` is fine. There is no support for FEM 
specific nodes such as `FacetNormal`, `jump`, `avg` and so on.

## Other alternatives
Expressions that represent UFL expressions (without substituting SymPy expressions for terminals) can compiled with 
[ufl-interpreter](https://github.com/MiroK/ufl-interpreter). This allows you to evaluate the UFL expressions which 
DOLFIN uses (e.g. to assemble forms) in arbitrary point in the finite element mesh.

[![Build Status](https://travis-ci.org/MiroK/ulfy.svg?branch=master)](https://travis-ci.org/MiroK/ulfy)
