from utils.utils import Counter


class ApproximateQAgent(object):
    """
       ApproximateQLearningAgent
    """
    def __init__(self):
        self.weights = Counter()

    def getWeights(self,state,action):

        return self.weights[(state, action)]

    def getQValue(self, state, action, features):
        """
          Should return Q(state,action) = w * featureVector
        """
        val = 0 
        for f in features:
          val += feats[f] * self.weights[f]
        return val

    def update(self, state, action, nextState, reward, features):
        """
           Should update your weights based on transition

           TODO: how to get argmax(nextQ)
        """
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action, features)
        
        for f in features: 
          self.weights[f] += self.alpha * features[f] * difference
