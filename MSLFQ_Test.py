'''
Created on Apr 8, 2013

@author: vima
'''
import unittest

from DummyEnvironment import DummyEnvironment
from MSLFQ import MSLFQ
import numpy as np

class Test(unittest.TestCase):

    def setUp(self):
        
        # Set up environment
        actions = ["Drug_A", "Drug_B"]
        num_features = 2
        gamma = 0.99
        stages = 1
        self.env = DummyEnvironment(feature_size=num_features, stages=1, gamma=gamma, actions=actions)
        
        #set up data
        
        self.data = []
        s = [1, 1]
        a = "Drug_A"
        r = 1.0
        ns = [0, 0]
        
        self.data.append([0, s, a, r, ns])
        
        s = [1, 1]
        a = "Drug_B"
        r = 0.0
        ns = [0, 0]
        
        self.data.append([0, s, a, r, ns])
        
        self.learner = MSLFQ(stages=stages, gamma=gamma)

    def testDummyEnvironment(self):
        state = [1.0, 1.0]
        action = "Drug_A"
        stage = 1
        features = self.env.phi(stage, state, action)
        
        result_list = [1.0, 1.0, 1.0, 0, 0, 0]
        self.assertListEqual(list(features), result_list)
        
        action = "Drug_B"
        result_list = [0, 0, 0, 1.0, 1.0, 1.0]
        features = self.env.phi(stage, state, action)
        self.assertListEqual(list(features), result_list)
        
    def testProcessDataFit(self):
        self.learner.process_data(self.data, self.env)
        
        n_data = self.learner.data_
        
        result_list = [1.0, 1.0, 1.0, 0, 0, 0]
        
        features = n_data[0][0]
        target = n_data[0][1]
        
        self.assertListEqual(list(features[0]), result_list)
        self.assertEqual(target[0], 1.0)
        
        result_list = [0, 0, 0, 1.0, 1.0, 1.0]
        self.assertListEqual(list(features[1]), result_list)
        self.assertEqual(target[1], 0.0)
        
        weights = self.learner.fit(features, target)
        
        self.assertEqual(np.sign(weights[0]), 1)
        self.assertEqual(np.sign(weights[3]), -1)
    
    def testLongFit(self):
        print self.learner.fit_data(self.data, self.env)
        
        
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
