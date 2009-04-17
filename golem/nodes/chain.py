class Chain:
  def __init__(self, nodes):
    assert(isinstance(nodes, list))
    self.nodes = nodes
      
  def train(self, d):
    for n in self.nodes:
      n.train(d)
      d = n.test(d)
    
  def test(self, d):
    for n in self.nodes:
      d = n.test(d)
    return d

  def __str__(self):
    return 'Chain (%s)' % ' -> '.join([str(n) for n in self.nodes])
