import logging

class BaseNode:
  def __init__(self, name='Node'):
    self.name = name
    self.empty_d = None

  @property
  def log(self):
    '''
    Logs are not deepcopy-able, so we need a property...
    '''
    return logging.getLogger('nodes.' + self.name)

  def train(self, d):
    self.log.info('training on ' + str(d))

    # store format of d
    self.empty_d = d[:0]

  def test(self, d):
    self.log.info('testing on ' + str(d))

    # check format of d
    if self.empty_d != None and self.empty_d != d[:0]:
      raise ValueError('Got %s, expected %s in %s' % 
        (self.empty_d, d, str(self)))

  def assert_two_class(self, d):
    '''Verify that dataset has exactly two classes'''
    if d.nclasses != 2:
      raise ValueError('Expected a DataSet with only 2 classes, got %d (%s)' %
        (d.ninstances, str(d.cl_lab)))

  def __str__(self):
    return self.name
