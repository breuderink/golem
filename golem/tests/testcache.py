import unittest, tempfile, cPickle, copy
from ..nodes.cache import DirCache, CacheNode
from ..nodes import PriorClassifier, RandomClassifier
from .. import data

class TestDirCache(unittest.TestCase):
  def setUp(self):
    self.file_cache = DirCache(tempfile.mkdtemp())
    for i in range(20):
      self.file_cache.add(i, range(i))

  def test_get(self):
    for i in range(20):
      self.assertEqual(self.file_cache.get(i), range(i))

  def test_key_error(self):
    self.assertRaises(KeyError, self.file_cache.get, -1)
    self.file_cache.add(-1, -1)
    self.assertEqual(self.file_cache.get(-1), -1)

  def test_persistence(self):
    cache_name = tempfile.mkdtemp()
    fc1 = DirCache(cache_name)
    for i in range(20):
      fc1.add(i, range(i))
    del fc1
    fc2 = DirCache(cache_name)
    for i in range(20):
      self.assertEqual(fc2.get(i), range(i))

class PickleMockNode(object):
  def __init__(self, nid=0):
    self.serialization_count = 0
    self.id = nid
    self.trained = False
    self.tested = False

  def train(self, d):
    self.trained = True
    return d

  def test(self, d):
    self.tested = True
    return d

  def __setstate__(self, state):
    self.__dict__ = state
    self.serialization_count += 1

  def __str__(self):
    return 'I was pickled %d times' % self.serialization_count


class TestCacheNode(unittest.TestCase):
  def setUp(self):
    self.d = data.gaussian_dataset([10, 10])
    self.d2 = data.gaussian_dataset([10, 10, 10])

  def test_training(self):
    cache_name = tempfile.mkdtemp()
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    self.failIf(cn.node.trained)

    # test first time
    cn.train(self.d)
    self.assert_(cn.node.trained)
    self.assertEqual(cn.node.serialization_count, 0)

    # test second time
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    d = cn.train(self.d)
    self.assertEqual(d, self.d)
    self.assert_(cn.node.trained)
    self.assertEqual(cn.node.serialization_count, 1)

    # test with different node
    n = PickleMockNode(nid=2)
    cn = CacheNode(n, cache_name) 
    d = cn.train(self.d)
    self.assertEqual(d, self.d)
    self.assert_(cn.node.trained)
    self.assertEqual(cn.node.serialization_count, 0)

    # test with different dataset
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    d2 = cn.train(self.d2)
    self.assertEqual(d2, self.d2)
    self.assert_(cn.node.trained)
    self.assertEqual(cn.node.serialization_count, 0)

  def test_testing(self):
    cache_name = tempfile.mkdtemp()
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    self.failIf(cn.node.tested)

    # test first time
    d = cn.test(self.d)
    self.assertEqual(d, self.d)
    self.assert_(cn.node.tested)
    self.assertEqual(cn.node.serialization_count, 0)

    # test second time
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    d = cn.test(self.d)
    self.assertEqual(d, self.d)
    self.assert_(cn.node.tested)
    self.assertEqual(cn.node.serialization_count, 1)

    # test with different node
    n = PickleMockNode(nid=2)
    cn = CacheNode(n, cache_name) 
    d = cn.test(self.d)
    self.assertEqual(d, self.d)
    self.assert_(cn.node.tested)
    self.assertEqual(cn.node.serialization_count, 0)

    # test with different dataset
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    d2 = cn.test(self.d2)
    self.assertEqual(d2, self.d2)
    self.assert_(cn.node.tested)
    self.assertEqual(cn.node.serialization_count, 0)


  def test_traintest(self):
    cache_name = tempfile.mkdtemp()
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    self.failIf(cn.node.trained)
    self.failIf(cn.node.tested)

    # test training, testing not cached
    cn.train(self.d)
    self.assert_(cn.node.trained)
    self.assertEqual(cn.node.serialization_count, 0)

    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    cn.test(self.d)
    self.assert_(cn.node.tested)
    self.assertEqual(cn.node.serialization_count, 0)

  def test_testtrain(self):
    cache_name = tempfile.mkdtemp()
    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    self.failIf(cn.node.trained)
    self.failIf(cn.node.tested)

    # test training, testing not cached
    cn.test(self.d)
    self.assert_(cn.node.tested)
    self.assertEqual(cn.node.serialization_count, 0)

    n = PickleMockNode()
    cn = CacheNode(n, cache_name) 
    cn.train(self.d)
    self.assert_(cn.node.trained)
    self.assertEqual(cn.node.serialization_count, 0)
