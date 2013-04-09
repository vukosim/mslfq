'''
Created on Apr 8, 2013

@author: Vukosi Marivate

'''


from sklearn import linear_model

import numpy as np

class MSLFQ(object):
    def __init__(self, stages=1, gamma=1):
        '''
        Constructor
        '''
        self.stages_ = stages
        self.gamma_ = gamma
    
    def process_data(self, data, env):
        self.data_ = {}
        
        for i in range(self.stages_):
            # [input_list,target_list,next_state]
            self.data_[i] = [[], [], []]
        for stage, s, a, r, ns in data:
            features = self.get_features(stage, s, a, env)
            self.data_[stage][0].append(features)
            self.data_[stage][1].append(r)
            if stage < self.stages_ - 1:
                [stage][2].append(ns)
    
    def update_targets(self, stage, weights, env):  
        sa_set = self.data_[stage][0]  
        r_set = self.data_[stage][1] 
        ns_set = self.data_[stage][2]
        
        for i in range(len(sa_set)):
            ns = ns_set[i]
            new_target = r_set[i] + self.gamma_ * (weights * env.phi(stage, ns, env.linear_policy(self, stage, weights, ns)))
            self.data_[stage][1][i] = new_target
        
    def fit_data(self, data, env):
        #Do the initial process
        self.process_data(data, env)
        range_ = range(self.stages_)
        self.weights_ = list(np.zeros(self.stages_))
        range_.sort(reverse=True)
        
        for stage in range_:
            self.weights_[stage] = self.fit(self.data_[stage][0], self.data_[stage][1])
            if (stage - 1) >= 0:
                self.update_targets(stage - 1, self.weights_[stage], env)
                
        return self.weights_
    
    def fit(self, input, target):
        clf = linear_model.LinearRegression()
        clf.fit(input, target)
        weights = clf.coef_
        return weights
    
    def get_features(self, stage, s, a, env):
        return env.phi(stage, s, a)
    
    def get_max_features(self, stage, s, env):
        w = self.weights_[stage]
        return env.phi(stage, s, env.linear_policy(self, stage, w, s))
