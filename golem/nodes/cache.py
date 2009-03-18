import cPickle, os, logging
from hashlib import sha1
from time import clock

class CacheNode:
  def __init__(self, node, cachedir):
    self.node = node
    self.cache = DirCache(cachedir)

  @classmethod
  def calc_hash(cls, d):
    return sha1(cPickle.dumps(d, cPickle.HIGHEST_PROTOCOL)).hexdigest()

  def log_duration(self, cached_time, uncached_time):
    saved_time = uncached_time - cached_time
    logging.getLogger('golem.nodes.Cache').info(
      'Cached. Saved %.2f seconds (from %.2f to %.2f).' % (
      saved_time, uncached_time, cached_time))
    if saved_time < 0:
      logging.getLogger('golem.nodes.Cache').warning(
      'Cashing took %.2f seconds longer! (from %.2f to %.2f).' % (
      -saved_time, uncached_time, cached_time))

  def train(self, d):
    '''
    Trains the node, but uses cached node and results if possible.
    (node_hash, d_hash) -> dict
    '''
    key = 'train' + CacheNode.calc_hash(d) + CacheNode.calc_hash(self.node)
    if self.cache.has(key):
      start_time = clock()
      value = self.cache.get(key)
      duration = clock() - start_time
      self.node = value['trained_node']
      d_out = value['d_out']

      # make user happy
      self.log_duration(value['duration'] - duration, value['duration'])
    else:
      start_time = clock()
      d_out = self.node.train(d) # node *can* change now!
      duration = clock() - start_time
      value = {'trained_node': self.node, 'd_out': d_out, 'duration': duration}
      self.cache.add(key, value)
    return d_out

  def test(self, d):
    '''
    Test using the node, but uses cached node and results if possible.
    (node_hash, d_hash) -> dict
    '''
    key = 'test' + CacheNode.calc_hash(d) + CacheNode.calc_hash(self.node)
    if self.cache.has(key):
      start_time = clock()
      value = self.cache.get(key)
      duration = clock() - start_time
      self.node = value['test_node']
      d_out = value['d_out']

      # make user happy
      self.log_duration(value['duration'] - duration, value['duration'])
    else:
      start_time = clock()
      d_out = self.node.test(d) # node *can* change now!
      duration = clock() - start_time
      value = {'test_node': self.node, 'd_out': d_out, 'duration': duration}
      self.cache.add(key, value)
    return d_out

class DirCache:
  def __init__(self, base_name):
    self.base_name = base_name
    try:
      f = open('%s_index.cache' % self.base_name, 'rb')
      self.index = cPickle.load(f)
      f.close()
    except IOError:
      self.index = {}

  def has(self, key):
    return key in self.index

  def get(self, key):
    if not self.has(key):
      raise KeyError
    value_fname = self.index[key]
    f = open(value_fname, 'rb')
    value = cPickle.load(f)
    f.close()
    return value

  def add(self, key, value):
    # write value to disk
    value_fname = '%s_%s.cache' % (self.base_name, 
      sha1(cPickle.dumps(key)).hexdigest())
    f = open(value_fname, 'wb')
    cPickle.dump(value, f, cPickle.HIGHEST_PROTOCOL)
    f.close()

    self.index[key] = value_fname
    f = open('%s_index.cache.new' % self.base_name, 'wb')
    cPickle.dump(self.index, f, cPickle.HIGHEST_PROTOCOL)
    f.close()

    # atomic update of index, does not work on windows.
    os.rename('%s_index.cache.new' % self.base_name,   
        '%s_index.cache' % self.base_name)

