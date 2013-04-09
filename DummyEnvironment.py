'''
@author: Vukosi Marivate
'''

import numpy as np

class DummyEnvironment(object):
    '''
    classdocs
    '''
    def __init__(self, feature_size=10, stages=1, gamma=0.95, actions=[0, 1]):
        '''
        Constructor
        '''
        feature_size = feature_size + 1
        self.gamma = gamma
        self.raw_actions = actions
        self.actions = actions
        self.num_actions = len(actions)
        self.stages = stages
        self.num_features = feature_size
        self.num_total_feature_vector = feature_size * self.num_actions
    
    def convert(self, state):
        features = list(state)
        features.append(1.0)
        return features
        
    def phi(self, stage, s, a):
        conv_state = self.convert(s)
        
        features = np.zeros(self.num_total_feature_vector)
        start = self.actions.index(a) * self.num_features
        stop = start + self.num_features 
        
        features[start:stop] = list(conv_state)
        return features

    def linear_policy(self, stage, w, s):
        # note that phi should be overridden (or learned in some way)
        a_index = np.argmax([np.dot(w, self.phi(stage, s, a)) for a in self.actions])
        return self.actions[a_index]
    
    def evaluate (self, stage, w, s, a):
        f = self.phi(s, a)
        #print f
        result = np.dot(w, f)
        return result
