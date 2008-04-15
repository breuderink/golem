from data import Data
from block import Block

class Group(Block):
  def __init__(self, block_list):
    Block.__init__(self)
    self.name = 'Group'
    self.model.block_list = block_list
      
  def train_method(self, data):
    block_list = self.model.block_list
    return [block.train(data) for block in block_list]
    
  def test_method(self, data):
   block_list = self.model.block_list
   return [block.test(data) for block in block_list]
    
  def __str__(self):
    return 'Group'
