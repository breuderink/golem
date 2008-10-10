class Group:
  def __init__(self, node_list):
    self.node_list = node_list
      
  def train(self, dataset):
    return [n.train(data) for n in self.node_list]
    
  def test(self, data):
    return [n.test(data) for n in self.node_list]

