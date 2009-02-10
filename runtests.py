#!/usr/bin/env python
import unittest
import tests
import logging

if __name__== '__main__':
  logging.basicConfig(level=logging.WARNING)
  suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
  #suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.TestDataSet)
  unittest.TextTestRunner(verbosity=1).run(suite)
