'''
@author: Vukosi Marivate
'''

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
        self.common_features = common
        self.stages = stages
        self.independent_features = feature_size - common
        self.num_features = feature_size
        self.num_total_feature_vector = stages * common + self.num_actions * self.independent_features * stages
    
    def convert(self, state):
        
        stage = state[0]
        #print state
        features = list(state[1:])
        features.append(1.0)
        return features, stage
        
        
    def phi(self, s, a, sparse=False, format="csr"):
        '''
        Calculates features
        '''
        if s[0] == -99999:
            conv_s = np.zeros(self.num_features)
            if sparse:
                cols = np.zeros(self.num_features + 1)
                start = self.actions.index(a) * self.num_features
                stop = start + self.num_features 
                indices = range(start, stop)
                indices.append(self.num_total_feature_vector - 1) 
                rows = np.array(indices)
                data = list(conv_s)
                data.append(0.0)
                data = np.array(data)
                sparse_features = sp_create_data(data, rows, cols, self.num_total_feature_vector, 1, format)
                return sparse_features
            else:
                features = np.zeros(self.num_total_feature_vector)
                start = self.actions.index(a) * self.num_features
                stop = start + self.num_features 
                features[start:stop] = list(conv_s)
                return features
        else:
            nfeatures, stage = self.convert(s)
            if self.common_features == 0:
                if sparse:
                    cols = np.zeros(self.num_features)
                    start = self.actions.index(a) * self.num_features + stage * self.num_actions * self.num_features
                    stop = start + self.num_features 
                    indices = range(start, stop)
                    rows = np.array(indices)
                    data = np.array(nfeatures)
                    sparse_features = sp_create_data(data, rows, cols, self.num_total_feature_vector, 1, format)
                    return sparse_features
                else:
                    features = np.zeros(self.num_total_feature_vector)
                    start = self.actions.index(a) * self.num_features + stage * self.num_actions * self.num_features
                    stop = start + self.num_features 
                    features[start:stop] = nfeatures
        return features

    def linear_policy(self, w, s):
        # note that phi should be overridden (or learned in some way)
        a_index = np.argmax([np.dot(w, self.phi(s, a)) for a in self.actions])
        return self.actions[a_index]
    
    def evaluate (self, w, s, a):
        f = self.phi(s, a)
        #print f
        result = np.dot(w, f)
        return result
    
def PickStartState(test_data, num):
    
    data = []
    
    for traj in test_data:
        for state, s, a, r, ns in traj:
            if s[0] == 0:
                data.append(s)
            if len(data) >= num:
                return data
            
if __name__ == '__main__':  
    history_file = "traj/hiv_training_data_3_stage"
    data = pickle.load(open(history_file))
    
    
    setup = SetupBatchHIV(data, num_test=300) 
    final_data, test_data, actions, stages, feature_size = setup
    
    test_start_states = PickStartState(test_data, 100)
    pickle.dump(test_start_states, open("traj/hiv_test_start_states", "wb"))
    
    
