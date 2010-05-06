import logging, warnings

class BaseNode:
  def __init__(self):
    self.name = self.__class__.__name__
    self.empty_d = None

    # test for overridden train and test methods:
    if self.__class__.train != BaseNode.train:
      raise Exception('Do not override methode train(). Use train_() instead.')
    if self.__class__.test != BaseNode.test:
      raise Exception('Do not override methode test(). Use test_() instead.')

  @property
  def log(self):
    '''
    Logs are not deepcopy-able, so we need a property...
    '''
    return logging.getLogger('golem.nodes.' + self.name)

  def train(self, d):
    self.log.info('training on ' + str(d))

    # store format of d
    self.empty_d = d[:0]

    # delegate call
    self.train_(d)

  @property
  def nclasses(self):
    return self.empty_d.nclasses

  def train_(self, d):
    '''
    Placeholder, meant to be replaced with the derived nodes train method.
    '''

  def test_(self, d):
    '''
    Placeholder, meant to be replaced with the derived nodes test method.
    '''
    pass

  def test(self, d):
    warnings.warn('Method [Node].test() is deprecated,' + 
      'use [Node].apply() instead.',
      DeprecationWarning)
    return self.apply(d)

  def apply(self, d):
    self.log.info('testing on ' + str(d))

    # check format of d
    if self.empty_d != None and self.empty_d != d[:0]:
      raise ValueError('Got %s, expected %s in %s' % 
        (d, self.empty_d, str(self)))

    # delegate call
    return self.test_(d)

  def train_apply(self, dtrain, dtest):
    '''
    Convenience method to train the node on dtrain, and apply it to dtest.
    '''
    self.train(dtrain)
    return self.apply(dtest)

  def __str__(self):
    return self.name
