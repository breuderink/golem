import unittest
import tests

if __name__== '__main__':
  suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
  unittest.TextTestRunner(verbosity = 1).run(suite)
