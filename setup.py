#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup

setup(name='mltools',
      version='0.1',
      description='some helper functions for machine learning',
      author='Gregor Sturm',
      author_email='mail@gregor-sturm.de',
      py_modules=['mltools.mltools', 'mltools.perfmeasures', 'mltools.weka'],
      install_requires=[
      ],
     )
