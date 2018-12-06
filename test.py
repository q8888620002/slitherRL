
import gym
import universe  # register the universe environments
import sys
import math
import numpy as np
import random 
import time
import pickle
import csv

from agent import ApproximateQAgent
from utils.env import create_slither_env
from universe.wrappers import Unvectorize


center_x = 270  
center_y = 235


# The snake moves to the directing of the mouse
# but to output the direction to a neural network we need to break the output to more discrete values

# the radius is the distance from the head of the snake to the mouse pointer(in pixel)
radius = 30
# This is the number of points we want around the head of the snake
# Ex: With 8 points where the mouse can be positioned around the head of the snake
# Note the distance from the point to the head is the same for all
#       *
#     *   *
#   *   s   *
#     *   *
#       *
# You can add more resolution to this if you want but it may increase learning time
resolution_points = 8
degree_per_slice = 360//resolution_points

# Available actions in the game
action_sheet = []
action_sheet_init = []

# We put all mouse positions in the action_sheet
for point in range(resolution_points):
    degree = point*degree_per_slice
    y_value_offset = radius * math.sin(math.radians(degree))
    x_value_offset = radius * math.cos(math.radians(degree))
    coord = universe.spaces.PointerEvent(center_x + x_value_offset, center_y + y_value_offset, 0)
    action_sheet.append((center_x + x_value_offset, center_y + y_value_offset))


    action_sheet_init.append(coord) 

### Convert observation numpy array into dictionary form with index 
### ["me_perc","snake_perc", "food_perc", "min_snake", "min_food"]
def dict_convert(features):

    new_features = dict()
    features_index = ['snake_dis', 'food_dis', 'snake_perc', 'food_perc', 'snake_50', 'snake_100']

    for i in range(len(features_index)):
      new_features[features_index[i]] = features[i]

    return new_features

if __name__ == '__main__':
 
  # Create customized and processed slither env
  #universe.configure_logging(False)
  
  env = create_slither_env('features')
  env = Unvectorize(env)

  env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

  observation_n = env.reset()

  ## init the q learning agent
  learning_agent = ApproximateQAgent()
  stored_weights = open('weights.pickle', 'rb')
  learning_agent.weights = pickle.load(stored_weights)
  ## randomly init an array
  action_coord = random.choice(action_sheet)

  start_time = []
  nextstate = 2

  coord=[]
  for point in range(8):
      degree = point*(360/8)
      y = 30 * math.sin(math.radians(degree))
      x = 30 * math.cos(math.radians(degree))
      coord.append((270+x, 235+y))


  for i_episode in range(200):
     # total reward of this episode
     total_reward = 0
     for t in range(1000):
       env.render() 
       action = universe.spaces.PointerEvent(action_coord[0],action_coord[1])
       observation_n, reward_n, done_n, info = env.step([action])
       total_reward += reward_n
       if reward_n == 0:
         reward_n = -1
       else:
         reward_n = reward_n * 2
       if done_n:
         reward_n = -5000

       features = dict_convert(observation_n)

       currentstate = nextstate
       nextstate = learning_agent.getAction(features)
       action_coord = coord[nextstate]
       print('action: ', action_coord)
       learning_agent.update(currentstate, reward_n, features)
       
       if done_n:
         print("Episode finished after {} timesteps".format(t+1))
         print((i_episode+1), "/", 200)

         ## update every episode
         stored_weights = open('weights.pickle', 'wb')
         pickle.dump(learning_agent.getWeights(), stored_weights)
         stored_weights.close()
         print("update weights")

         # write to csv file
         myData = [[1, 2, 3], ['Good Morning', 'Good Evening', 'Good Afternoon']]  
         myFile = open('output.csv', 'a')  
         with myFile: 
          writer = csv.writer(myFile)
          data = [i_episode+1, total_reward]
          for f in learning_agent.features:
            data.append(learning_agent.weights[f])
          writer.writerow(data)

         break

  env.close() 

