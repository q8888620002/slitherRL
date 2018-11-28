import gym
import universe  # register the universe environments
import sys
import math
import numpy as np

from utils.env import create_slither_env
from universe.wrappers import Unvectorize


center_x = 270
center_y = 235
# Game screen corners
# ul_x = 20
# ul_y = 85
# lr_x = 520
# lr_y = 385
# The above bound might be too big so I just +-100 the center_x and center_y
ul_x = 170
ul_y = 135
lr_x = 370
lr_y = 335

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

# We put all mouse positions in the action_sheet
for point in range(resolution_points):
    degree = point*degree_per_slice
    x_value_offset = radius * math.sin(math.radians(degree))
    x_value_offset = radius * math.cos(math.radians(degree))
    coord = universe.spaces.PointerEvent(center_x + x_value_offset, center_y + x_value_offset, 0)
    action_sheet.append(coord)

if __name__ == '__main__':

  # Create customized and processed slither env
  #universe.configure_logging(False)
  env = create_slither_env('features')

  env = Unvectorize(env)
  env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

  observation_n = env.reset()

  print("Num_actions: " + str(env.action_space.n))


  while True:
    action_n = env.action_space.sample()
    print(observation_n.shape)

    observation_n, reward_n, done_n, info = env.step(action_n)
    print(observation_n)
    if sys.platform == 'linux':
      #Cant render on server
      print(observation_n)
      print(observation_n.shape)
    else:
      env.render()
