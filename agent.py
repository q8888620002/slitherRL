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
        #       *
        #     *   *
        #   *   s   *
        #     *   *
        #       *
        self.features = ["me_perc","snake_perc", "food_perc", "min_snake", "min_food", "food_leftTop", "food_rightTop", "food_leftBottom", "food_rightBottom"]
        self.resolution_points = 8
        self.degree_per_slice = 360/self.resolution_points

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

        for a in self.actions:
            self.weights[a] = Counter()
            for f in range(len(self.features)):
              self.weights[a][f] = 1

        for a in self.actions:
          self.weights[a][4] = 100
          self.weights[a][5] = 100
          self.weights[a][6] = 100
          self.weights[a][7] = 100
          self.weights[a][8] = 100

    def getQValue(self, action, features):
        """
          Should return Q(state,action) = w * featureVector
        """
        val = 0 
        for f in self.features:
          val += features[f] * self.weights[action][f]
        return val

    def getWeight(self):
        for a in self.actions:
          print("state:  ", a)
          for f in self.features:
            print(f, self.weights[a][f])
        ## Return the arg_max q value of next state 

    def getMaxQ(self, features):
        max_a =  max(self.actions, key = lambda a: self.getQValue(a, features))
        return self.getQValue(max_a, features)


    def update(self, action, reward, features, done):
        """
           Should update the weights based on transition
        """
        reward_n = reward
        if done:
          reward_n = -100
        else:
          if reward == 0:
            reward_n = reward * 10

        difference = reward_n + self.discount * self.getMaxQ(features) - self.getQValue(action, features)
        for f in features: 
          self.weights[action][f] += self.alpha * features[f] * difference

      ### Return the best action according to the current feature
    def getAction(self, features):
        # return the action with the max q value
        return max(self.actions, key = lambda a: self.getQValue(a, features))





