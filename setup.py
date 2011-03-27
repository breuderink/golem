#!/usr/bin/env python
from distutils.core import setup

setup(name='golem',
  version='0.2',
  license='BSD',
  url='http://www.borisreuderink.nl',
  author='Boris Reuderink',
  author_email='b.reuderink@gmail.com',
  packages=['golem', 'golem.nodes', 'golem.tests'],
  )
