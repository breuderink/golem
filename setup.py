from distutils.core import setup

setup(name='golem',
  version='0.1',
  license='GPL',
  url='http://www.borisreuderink.nl',
  author='Boris Reuderink',
  author_email='b.reuderink@gmail.com',
  packages=['golem', 'golem.nodes', 'golem.tests'],
  py_modules=['__init__', 'runtests']
  )
