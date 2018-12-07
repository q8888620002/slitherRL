from utils.utils import Counter
import universe  # register the universe environments
import utils.utils as utils
import math

class ApproximateQAgent(object):
    """
       ApproximateQLearningAgent
    """
    def __init__(self):
        self.weights =dict()

        self.center_x = 270
        self.center_y = 235
        self.radius = 30
        self.n_area = 12
        # This is the number of points we want around the head of the snake
        # Ex: With 8 points where the mouse can be positioned around the head of the snake
        # Note the distance from the point to the head is the same for all
        #       2
        #       *
        #   3 *   * 1
        # 4 *   s   * 0
        #   5 *   * 7
        #       *
        #       6
        self.features = ['snake_dis', 'food_dis', 'snake_perc', 'food_perc', 'snake_50', 'snake_100', 'neighbor_snake_dis', 'neighbor_snake_per', 'neighbor_food_dis', 'neighbor_food_per']
        self.degree_per_slice = 360/self.n_area

        # Available actions in the game
        self.actions = utils.create_actionList(self.n_area, self.radius)
        self.discount = 0.8
        self.alpha = 0.5
        self.weights = {'snake_dis': 4, 
                        'food_dis': -2, 
                        'snake_perc': -4, 
                        'food_perc': 2, 
                        'snake_50': -10,
                        'snake_100': -10,
                        'neighbor_snake_dis': 2, 
                        'neighbor_snake_per': -2, 
                        'neighbor_food_dis': 1, 
                        'neighbor_food_per': -1}

    def getQValue(self, action, features):
        """
          Should return Q(state,action) = w * featureVector
        """
        val = 0 
        for f in features:
          val += features[f].flatten()[action] * self.weights[f]
        return val

    def getWeights(self):
         return self.weights

    def getMaxQ(self, features):
        max_q = -float('inf')
        for a in range(8):
          new_q = self.getQValue(a, features)
          if new_q > max_q:
            max_q = new_q

        return max_q


    def update(self, action, reward, features):
        """
           Should update the weights based on transition
        """
        difference = reward + self.discount * self.getMaxQ(features) - self.getQValue(action, features)
        for f in features: 
            self.weights[f] += self.alpha * features[f].flatten()[action] * difference
        print('weight:', self.weights)

    ### Return the best action according to the current feature
    def getAction(self, features):
        action = 0
        max_q = -float('inf')
        for a in range(self.n_area):
          new_q = self.getQValue(a, features)
          print('===', a, '===', new_q)
          if new_q > max_q:
            max_q = new_q
            action = a 
        return action
