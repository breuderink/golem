from distutils.core import setup

setup(name='Golem',
  version='0.1',
  license='GPL',
  url='www.borisreuderink.nl',
  author='Boris Reuderink',
  author_email='b.reuderink@gmail.com',
  packages=['golem', 'golem.nodes', 'tests'],
  py_modules=['runtests']
  )
