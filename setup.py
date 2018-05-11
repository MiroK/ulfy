#!/usr/bin/env python

from distutils.core import setup

setup(name = 'ufl_mms',
      version = '0.1',
      description = 'Compile DOLFIN Expressions from SymPy expressions',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/ufl_mms.git',
      packages = ['ufl_mms'],
      package_dir = {'ufl_mms': 'ufl_mms'}
)
