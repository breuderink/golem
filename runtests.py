#!/usr/bin/env python

def test():
  import unittest
  import golem.tests
  import logging
  logging.basicConfig(level=logging.WARNING)
  suite = unittest.defaultTestLoader.loadTestsFromModule(golem.tests)
  unittest.TextTestRunner(verbosity=1).run(suite)

if __name__=='__main__':
  test()
