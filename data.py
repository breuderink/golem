from types import *

# @@TODO: make fields write protected
# Current implementation loses trace. Should the trace be decoupled from Data?

class Data:
  def __init__(self, xs, ys, signature):
    assert(type(xs) == type(ys) == ListType)
    assert(type(xs[0]) == type(ys[0]) == ListType)
    assert(len(xs) == len(ys))
    self.xs = xs
    self.ys = ys
    self.ninstances = len(xs)
    self.signature = signature[:]
  
  # make read-only properties
  def __setattr__(self, name, val):
    if not name in ['xs', 'ys', 'ninstances'] or not name in self.__dict__:
      # write once
      self.__dict__[name] = val
    else:
      raise AttributeError, '%s is read only' % (name)


  def add_signature(self, name):
    self.signature.append(name)
    
  def __str__(self):
    state_str = 'Data (%d instances, %d features, %dD labels)' % \
      (len(self.xs), len(self.xs[0]), len(self.ys[0]))
    return ' -> '.join(self.signature + [state_str])
     
