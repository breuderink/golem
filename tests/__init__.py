__all__ = ['testdataset', 'testartificialdata', 'testkernel', 'testsvm', 
  'testloss', 'testcrossvalidation', 'testonevsrest', 'testzscore']

from testdataset import TestDataSet, TestDataSetConstruction
from testartificialdata import TestGaussianData
from testkernel import TestKernel
from testsvm import TestSVM, TestSVMPlot
from testloss import TestLoss
from testcrossvalidation import TestCrossValidation
from testonevsrest import TestOneVsRest
from testzscore import TestZScore
