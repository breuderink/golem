import unittest, tempfile, cPickle
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

class TestCacheNode(unittest.TestCase):
  def setUp(self):
    self.d = data.gaussian_dataset([10, 10])

  def test_train(self):
    cache_name = tempfile.mkdtemp()
    n0 = PriorClassifier()
    n1 = PriorClassifier()
    cn1 = CacheNode(n1, cache_name)

    # test construction
    self.assertEqual(id(cn1.node), id(n1))
    self.assertEqual(cPickle.dumps(cn1.node), cPickle.dumps(n0))

    # test that training modifies node
    d_out1 = cn1.train(self.d)
    self.failIfEqual(cPickle.dumps(cn1.node), cPickle.dumps(n0))
    self.assertEqual(id(cn1.node), id(n1))

    # create second CacheNode
    n2 = PriorClassifier()
    cn2 = CacheNode(n2, cache_name)
    self.assertEqual(cPickle.dumps(cn2.node), cPickle.dumps(n0))

    # test cached training
    d_out2 = cn2.train(self.d)
    self.failIfEqual(id(cn2.node), id(n2))
    self.assertEqual(d_out1, d_out2)
    self.assertEqual(cPickle.dumps(cn1.node), cPickle.dumps(cn2.node))

  def test_test(self):
    cache_name = tempfile.mkdtemp()
    n0 = RandomClassifier()
    n1 = RandomClassifier()

    # fill cache
    cn1 = CacheNode(n1, cache_name)
    cn1.train(self.d)
    d_out1 = cn1.test(self.d)

    # create second CacheNode
    n2 = RandomClassifier()
    cn2 = CacheNode(n1, cache_name)
    cn2.train(self.d)
    d_out2 = cn2.test(self.d)

    # compare results
    self.assertEqual(cPickle.dumps(d_out1), cPickle.dumps(d_out2))
