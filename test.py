
import gym
import universe  # register the universe environments
import sys
import math
import numpy as np
import random 
import time
import datetime
import pickle
import utils.utils as utils

from agent import ApproximateQAgent
from utils.env import create_slither_env
from universe.wrappers import Unvectorize

# center of the frame
center_x = 270  
center_y = 235

if __name__ == '__main__':
 
  # Create customized and processed slither env
  #universe.configure_logging(False)
  
  env = create_slither_env('features')
  env = Unvectorize(env)
  env.configure(fps=10.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

  observation_n = env.reset()

  ## init the q learning agent
  learning_agent = ApproximateQAgent()
  
  # read in stored weight from previous games with pickle
  #learning_agent.weights = pickle.load(open('weights.pickle', 'rb'))

  # create coord_list with (36 coordinates, 30 radius), randomly move at the first actions
  coord = utils.create_actionList(12, 30)
  action_coord = random.choice(coord)
  nextstate = 2

  for i_episode in range(200):
     # total reward of this episode
     total_reward = 0
     for t in range(1000):
       env.render() 
       start_time = datetime.datetime.now()

       action = universe.spaces.PointerEvent(action_coord[0],action_coord[1])
       observation_n, reward_n, done_n, info = env.step([action])

       # add game returned reward to output, change agent update reward to custom reward
       total_reward += reward_n
       reward = utils.redefine_reward(reward_n, done_n)

       # convert features into dictionary
       features = utils.dict_convert(observation_n, learning_agent.features)

       currentstate = nextstate
       nextstate = learning_agent.getAction(features)
       action_coord = coord[nextstate]
       print('action: ', action_coord)
       learning_agent.update(currentstate, reward, features)
       
       if done_n:
         print("Episode finished after {} timesteps".format(t+1))
         print((i_episode+1), "/", 200)
         done_time = datetime.datetime.now()
         ## update every episode
         utils.dump_to_pickle('weights.pickle', learning_agent.getWeights())
         output = [i_episode+1, total_reward, start_time, done_time, done_time - start_time, t] 
         for f in learning_agent.features:
            output.append(learning_agent.weights[f])
         # write output to csv file
         utils.append_to_csv('output_10_features.csv', output)
         break

  env.close() 

