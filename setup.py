#!/usr/bin/env python

from distutils.core import setup

setup(name = 'ulfy',
      version = '0.1',
      description = 'Translate UFL to Sympy to Expression',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/ulfy.git',
      packages = ['ulfy'],
      package_dir = {'ulfy': 'ulfy'}
)
