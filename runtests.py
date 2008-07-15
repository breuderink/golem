import unittest

from tests import *

if __name__== '__main__':
  suite = unittest.TestSuite()
  suite.addTest(testdataset.suite())
  suite.addTest(testartificialdata.suite())
  unittest.TextTestRunner(verbosity = 2).run(suite)
