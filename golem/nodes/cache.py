import cPickle, os, logging
from hashlib import sha1
from time import clock

class Cache:
  def __init__(self, node, cachedir):
    self.node = node
    self.cache = DirCache(cachedir)

  @classmethod
  def calc_hash(cls, d):
    return sha1(cPickle.dumps(d, cPickle.HIGHEST_PROTOCOL)).hexdigest()

  def log_duration(self, cached_time, uncached_time):
    saved_time = uncached_time - cached_time
    if saved_time > 0:
      gain = saved_time / (uncached_time + 1e-8)
      logging.getLogger('golem.nodes.Cache').info(
        'Caching saved %.2f seconds (%.1f%%).' % (
        saved_time, gain * 100.))
    elif saved_time == 0:
      logging.getLogger('golem.nodes.Cache').info(
        'Caching saved %.2f seconds.' % saved_time)
    else:
      logging.getLogger('golem.nodes.Cache').warning(
        'Caching took %.2f seconds (from %.2f to %.2f).' % (
        saved_time, uncached_time, cached_time))

  def train(self, d):
    '''
    Trains the node, but uses cached node and results if possible.
    (node_hash, d_hash) -> dict
    '''
    key = 'train' + d.hash() + Cache.calc_hash(self.node)
    return self.cached_call(key, self.node.train, d)

  def test(self, d):
    '''
    Test using the node, but uses cached node and results if possible.
    (node_hash, d_hash) -> dict
    '''
    key = 'test' + d.hash() + Cache.calc_hash(self.node)
    func = self.node.test
    return self.cached_call(key, self.node.test, d)

  def cached_call(self, key, func, d):
    if self.cache.has(key):
      start_time = clock()
      value = self.cache.get(key)
      duration = clock() - start_time
      self.node = value['node']
      d_out = value['d_out']

      # make user happy
      self.log_duration(duration, value['duration'])
    else:
      start_time = clock()
      d_out = func(d) # node *can* change now!
      duration = clock() - start_time
      value = {'node': self.node, 'd_out': d_out, 'duration': duration}
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

    # safe update
    try:
      os.remove('%s_index.cache' % self.base_name)
    except OSError:
      pass
    os.rename('%s_index.cache.new' % self.base_name,   
        '%s_index.cache' % self.base_name)
