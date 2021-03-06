import logging, warnings

class BaseNode:
  def __init__(self):
    self.name = self.__class__.__name__
    self.empty_d = None

    # test for overridden train and test methods:
    if self.__class__.train != BaseNode.train:
      raise Exception('Do not override methode train(). Use train_() instead.')
    if self.__class__.test != BaseNode.test:
      raise Exception('Do not override methode test(). Use apply_() instead.')
    if self.__class__.test_ != BaseNode.test_:
      raise Exception('Do not override methode test_(). Use apply_() instead.')
    if self.__class__.apply != BaseNode.apply:
      raise Exception('Do not override methode apply(). Use apply_() instead.')

  @property
  def log(self):
    '''
    Logs are not deepcopy-able, therefore we need a property...
    '''
    return logging.getLogger('golem.nodes.' + self.name)

  @property
  def nclasses(self):
    return self.empty_d.nclasses

  def test(self, d):
    warnings.warn('Method [Node].test() is deprecated, ' + 
      'use [Node].apply() instead.',
      DeprecationWarning)
    return self.apply(d)

  def train(self, d):
    self.log.info('training on ' + str(d))

    # store format of d
    self.empty_d = d[:0]

    # delegate call
    self.train_(d)

  def apply(self, d):
    self.log.info('testing on ' + str(d))

    # check validity of d
    if d.ninstances == 0:
      raise ValueError('Got empty DataSet.')

    if self.empty_d!= None and self.empty_d != d[:0]:
      raise ValueError('Got %s, expected %s in %s' % 
        (d, self.empty_d, str(self)))

    # delegate call
    return self.apply_(d)

  def train_apply(self, dtrain, dtest):
    '''
    Convenience method to train the node on dtrain, and apply it to dtest.
    '''
    self.train(dtrain)
    return self.apply(dtest)

  def __str__(self):
    return self.name

  def train_(self, d):
    '''
    Placeholder, meant to be replaced with the derived nodes train method.
    '''

  def apply_(self, d):
    '''
    Placeholder, meant to be replaced with the derived nodes apply method.
    '''
    pass

  def test_(self, d):
    '''Deprecated '''
    pass
