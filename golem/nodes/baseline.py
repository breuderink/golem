import numpy as np
from ..dataset import DataSet
from basenode import BaseNode

class PriorClassifier(BaseNode):
  """
  Classifier node that bases the classification on how many instances
  of each class are in the training data.
  
  For each instance ys (labels) contains a chance value for each 
  possible class. The xs (samples with features) returned by this node 
  will also adhere to this format: 
  per sample, for each class, a chance value [0.0, 0.1].
  
  To compare the classification results of different data sets, you
  can use the performance functions in golem.perf. 
  
  For example::
    
    import golem
    import numpy as np
    
    # Generate the data.
    xs = np.random.rand(100,6) # 100 samples, 6 features
    ys = np.random.rand(100,2) # 100 samples, 2 classes
    ys = golem.helpers.hard_max(ys) # hard labels
    d = golem.DataSet(xs, ys, None)
    trainingset = d[:50] # use first half as training
    testset = d[50:] # second half as test set
    
    # Create, train, and apply the classifiers.
    classifier1 = golem.nodes.PriorClassifier()
    classifier2 = golem.nodes.SVM()    
    d1 = classifier1.train_apply(trainingset, testset)
    d2 = classifier2.train_apply(trainingset, testset)
    
    # Show the performance of the classifiers. 
    print 'Prior Classifier Accuracy:', golem.perf.accuracy(d1)
    print 'SVM Classifier Accuracy:', golem.perf.accuracy(d2)
    
  These baseline classifiers (Prior, Random, and WeakClassifier)
  provide a nice way to compare the results that you get to these
  situations. Best to use in combination with (repeated) Cross 
  Validation (golem.cv.cross_validate) for a more steady result.
  """
  def train_(self, d):
    """
    Determines the class that has the most instances in the training 
    data, and stores this for use by apply().
    """
    self.mfc = np.argmax(d.ninstances_per_class)

  def apply_(self, d):
    """
    Returns for each sample a certain (1.0) classification for the 
    class that had the highest number of instances in the training
    data.
    """
    xs = np.zeros(d.ys.shape)
    xs[:, self.mfc] = 1
    return DataSet(xs, d.ys, d.ids, cl_lab=d.cl_lab)

  def __str__(self): 
    if hasattr(self, 'mfc'):
      return 'PriorClassifier (class=%d)' % self.mfc
    else:
      return 'PriorClassifier'

class RandomClassifier(BaseNode):
  """
  Classifier node that generates random values between [0.0, 1.0).
  
  See the documentation for PriorClassifier for more information 
  on ys, xs, and on how to use such a classifier node.
  """
  def apply_(self, d):
    return DataSet(xs=np.random.random(d.ys.shape), default=d)

class WeakClassifier(BaseNode):
  """
  This is a simulation of a weak classifier. It generates random output,
  with a slight bias towards the true labels. 
  *THE TRUE LABELS ARE USED IN THE TEST METHOD*.
    
  See the documentation for PriorClassifier for more information 
  on ys, xs, and on how to use such a classifier node.
  """
  def apply_(self, d):
    return DataSet(xs=np.random.random(d.ys.shape) + .10 * d.ys, default=d)
