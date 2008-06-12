from types import *

class Node:
  def __init__(self):
    self.hyperparams = {}
    self.name = 'Basic node'
    self.model = Model()
    self.is_initialized = True

  def train(self, data, trace):
    print 'Training %s...' % str(self)
    return self.add_trace(self.train_method(data))
  
  def test(self, data, trace):
    print 'Testing %s...' % str(self)
    return self.add_trace(self.test_method(data))
       
  def train_method(self, data):
    '''to be overridden'''
    return self.test_method(data)

  def test_method(self, data):
    '''to be overridden'''
    return data
    
  def add_trace(self, data):
    #@@ADDME @@FIXME
    if type(data) == ListType:
      for d in data:
        d.add_signature(str(self))
    else:
      data.add_signature(str(self))
    return data
  
  def __str__(self):
    hyperparams = self.hyperparams
    if len(hyperparams) > 0:
      return '%s %s' % (self.name, str(hyperparams))
    else:
      return self.name
