from utils.utils import Counter
import universe  # register the universe environments
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
        self.features = ['snake_dis', 'food_dis', 'snake_perc', 'food_perc', 'snake_50', 'snake_100']
        self.resolution_points = 8
        self.degree_per_slice = 360//self.resolution_points

        # Available actions in the game
        self.actions = []
        self.discount = 0.8
        self.alpha = 0.5
        # We put all mouse positions in the action_sheet
        for point in range(self.resolution_points):
            degree = point*self.degree_per_slice
            y_value_offset = self.radius * math.sin(math.radians(degree))
            x_value_offset = self.radius * math.cos(math.radians(degree))
            coord = universe.spaces.PointerEvent(self.center_x + x_value_offset, self.center_y + y_value_offset, 0)
            self.actions.append((self.center_x + x_value_offset, self.center_y + y_value_offset))

        self.weights = {'snake_dis': 1, 
                        'food_dis': 5, 
                        'snake_perc': -1, 
                        'food_perc': 5, 
                        'snake_50': 1000,
                        'snake_100': 50}

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
          #if f !='danger_snake':
            self.weights[f] += self.alpha * features[f].flatten()[action] * difference
        print('weight:', self.weights)

      ### Return the best action according to the current feature
    def getAction(self, features):
        action = 0
        max_q = -float('inf')
        for a in range(8):
          new_q = self.getQValue(a, features)
          print('===', a, '===', new_q)
          if new_q > max_q:
            max_q = new_q
            action = a 
        return action
