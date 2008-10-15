import unittest

from tests import *
import tests

if __name__== '__main__':
  suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
  unittest.TextTestRunner(verbosity = 2).run(suite)
