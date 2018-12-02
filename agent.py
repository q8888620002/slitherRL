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
        self.features = ["me_perc","snake_perc", "food_perc", "min_snake", "min_food"]
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

        for a in self.actions:
            self.weights[a] = Counter()

    def getQValue(self, action, features):
        """
          Should return Q(state,action) = w * featureVector
        """
        val = 0 
        print(features)
        for f in self.features:
          val += features[f] * self.weights[action][f]
        return val


        ## Return the arg_max q value of next state 

    def getMaxQ(self, features):
        max_q = 0 

        for a in self.actions:
          new_q = self.getQValue(a, features)
          if new_q > max_q:
            max_q = new_q

        return max_q


    def update(self, action, reward, features):
        """
           Should update your weights based on transition
           TODO: how to get argmax(nextQ)
        """
        difference = reward + self.discount * self.getMaxQ(features) - self.getQValue(action, features)
        
        for f in features: 
          self.weights[action][f] += self.alpha * features[f] * difference

      ### Return the best action according to the current feature
    def getAction(self, features):
        action = None
        max_q = 0
        for a in self.actions:
          new_q = self.getQValue(a, features)
          if new_q > max_q:
            max_q = new_q
            action = a 
        return a





