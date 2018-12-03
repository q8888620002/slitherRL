import gym
import universe  # register the universe environments
import numpy as np
import math
import pyglet
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from gym.spaces.box import Box
from gym import spaces
from collections import deque

from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe.wrappers.experimental import SafeActionSpace
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode

class SimpleImageViewer(object):
  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display)
      self.width = width
      self.height = height
      self.isopen = True

    nchannels = arr.shape[-1]
    if nchannels == 1:
      _format = "I"
    elif nchannels == 3:
      _format = "RGB"
    else:
      raise NotImplementedError
    image = pyglet.image.ImageData(self.width, self.height, "RGB", arr.tobytes())

    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

class CropScreen(vectorized.ObservationWrapper):
  def __init__(self, env, height, width, top=0, left=0):
    """Crops out a [height]x[width] area starting from (top,left) """
    super(CropScreen, self).__init__(env)
    self.height = height
    self.width = width
    self.top = top
    self.left = left
    self.observation_space = Box(0, 255, shape=(height, width, 3))

  def _observation(self, observation_n):
    return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None for ob in observation_n]

class FixedKeyState(object):
  def __init__(self, keys):
    self._keys = [keycode(key) for key in keys]
    self._down_keysyms = set()

  def apply_vnc_actions(self, vnc_actions):
    for event in vnc_actions:
      if isinstance(event, vnc_spaces.KeyEvent):
        if event.down:
          self._down_keysyms.add(event.key)
        else:
          self._down_keysyms.discard(event.key)

  def to_index(self):
    action_n = 0
    for key in self._down_keysyms:
      if key in self._keys:
        # If multiple keys are pressed, just use the first one
        action_n = self._keys.index(key) + 1
        break
    return action_n

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
  def __init__(self, env, keys):
    super(DiscreteToFixedKeysVNCActions, self).__init__(env)

    self._keys = keys
    self._generate_actions()
    self.action_space = spaces.Discrete(len(self._actions))

  def _generate_actions(self):
    self._actions = []
    uniq_keys = set()
    for key in self._keys:
      for cur_key in key.split(' '):
        uniq_keys.add(cur_key)

    for key in [''] + self._keys:
      split_keys = key.split(' ')
      cur_action = []
      for cur_key in uniq_keys:
        cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
      self._actions.append(cur_action)
    self.key_state = FixedKeyState(uniq_keys)

  def _action(self, action_n):
    # Each action might be a length-1 np.array. Cast to int to avoid warnings.
    return [self._actions[int(action)] for action in action_n]

class RenderWrapper(vectorized.Wrapper):
  def __init__(self, env, state_type):
    self.viewer = None

    self.state_type = state_type
    self.processor_colors = SlitherProcessor('colors')
    self.processor_shapes = SlitherProcessor('shapes')
    self.processor_features = SlitherProcessor('features')

    self.state_size = self.processor_shapes.state_size
    self.high_val = self.processor_shapes.high_val

    super(RenderWrapper, self).__init__(env)

  def _reset(self):
    self.orig_obs = self.env.reset()
    self.proc_obs_shapes = self.processor_shapes.process(np.copy(self.orig_obs))
    self.proc_obs_colors = self.processor_colors.process(np.copy(self.orig_obs))
    self.proc_obs_features = self.processor_features.process(np.copy(self.orig_obs))

    small_proc_obs = self.processor_features.resize(np.copy(self.proc_obs_colors))
    return small_proc_obs

  def _step(self, action):
    self.orig_obs, reward, done, info = self.env.step(action)
    self.proc_obs_shapes = self.processor_shapes.process(np.copy(self.orig_obs))
    self.proc_obs_colors = self.processor_colors.process(np.copy(self.orig_obs))
    self.proc_obs_features = self.processor_features.process(np.copy(self.orig_obs))

    small_proc_obs = self.processor_features.resize(np.copy(self.proc_obs_features))
    return small_proc_obs, reward, done, info

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.state_type == 'shapes':
      gray_img = self.proc_obs_shapes[0]*100
      proc_obs = np.concatenate((gray_img, gray_img, gray_img),2)
      img = np.concatenate((self.orig_obs[0][::-1,:,:],proc_obs[::-1,:,:],self.proc_obs_colors[0][::-1,:,:]),1)

    elif self.state_type == 'colors':
      img = np.concatenate((self.orig_obs[0][::-1,:,:],self.proc_obs_colors[0]),1)

    elif self.state_type =='features':
      img = np.concatenate((self.orig_obs[0][::-1,:,:],self.proc_obs_colors[0][::-1,:,:]),1)

    else:
      img = self.orig_obs[0][::-1,:,:]

    img = img.astype(np.uint8)
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = SimpleImageViewer()
      self.viewer.imshow(img)

class SlitherProcessor(object):
  def __init__(self, state_type):
    self.state_type = state_type

    if self.state_type == 'features':
      self.state_size = [5,1,1]
      self.zoom = (1,1,1)
      self.high_val = 1.0

    elif self.state_type == 'colors':
      self.state_size = [30,50,3]
      self.zoom = (.1,.1,1)
      self.high_val = 255.0

    elif self.state_type == 'shapes':
      self.state_size = [30,50,1]
      self.zoom = (.1,.1,1)
      self.high_val = 1.0

    elif self.state_type == 'transfer':
      self.state_size = [224,224,3]
      self.zoom = (.75,.75,1)
      self.high_val = 255.0

    else: NotImplementedError

  def process(self, frames):
    if self.state_type == 'features':
      return [self.process_features(f) for f in frames]

    elif self.state_type == 'colors':
      return [self.process_colors(f) for f in frames]

    elif self.state_type == 'shapes':
      return [self.process_shapes(f) for f in frames]

    elif self.state_type == 'transfer':
      frames = [self.process_colors(f) for f in frames]
      return [f[1:,101:-100,:] for f in frames]

  def resize(self, frames):
    if self.state_type != 'features':
      return [ndimage.zoom(f, self.zoom, order=2) for f in frames]
    else:
      return frames

  def process_shapes(self,frame):
    frame = self.remove_background(frame)
    label = self.connected_components(frame)
    return self.extract_shapes(label)

  def process_colors(self,frame):
    frame = self.remove_background(frame)
    label = self.connected_components(frame)
    return self.extract_colors(frame,label)

  def process_features(self,frame):
    frame = self.remove_background(frame)
    label = self.connected_components(frame)
    frame = self.extract_colors(frame,label)
    return self.extract_features(frame)

  def remove_background(self,frame):
    abs_t = 115
    frame[(frame[:,:,0]<abs_t)*(frame[:,:,1]<abs_t)*(frame[:,:,2]<abs_t)] = 0

    rel_t = 30
    avg_pix = np.mean(frame,2)
    diff = np.abs(avg_pix-frame[:,:,0]) + np.abs(avg_pix-frame[:,:,1]) + np.abs(avg_pix-frame[:,:,2])
    frame[:,:,:] = 255
    frame[diff<rel_t] = 0
    return frame

  def connected_components(self, frame):
    sing_frame = ndimage.grey_erosion(frame[:,:,1], size=(2,2))
    blur_radius = .35
    sing_frame = ndimage.gaussian_filter(sing_frame, blur_radius)
    labeled, self.nr_objects = ndimage.label(sing_frame)
    return labeled[:,:,np.newaxis]

  def extract_shapes(self, frame):
    frame[frame != 0] = 1
    return frame

  def extract_colors(self, frame, label):
    label = label[:,:,0]
    snake_threshold = 235
    enemy_c = [255,0,0]
    me_c = [0,255,0]
    food_c = [0,0,255]
    frame[:,:,:] = 0
    me_label = np.bincount(label[145:155,245:255].flatten().astype(int))[1:]
    if len(me_label)>0:
      me_label = np.argmax(me_label) + 1
    else:
      me_label = -1
    for i in range(self.nr_objects):
      cur_label = i+1
      size = np.count_nonzero(label[label==cur_label])
      if size<snake_threshold:
        frame[label==cur_label] = food_c
      elif me_label==cur_label:
        frame[label==cur_label] = me_c
      else:
        frame[label==cur_label] = enemy_c
    return frame

  def extract_features(self, frame):
    snake_layer = frame[:,:,0]
    me_layer = frame[:,:,1]
    food_layer = frame[:,:,2]

    num_pix = frame.shape[0]*frame.shape[1]
    max_dis = 150+250

    snake_pix = np.count_nonzero(snake_layer)
    me_pix = np.count_nonzero(me_layer)
    food_pix = np.count_nonzero(food_layer)


    snake_perc = (num_pix - snake_pix)*1.0/num_pix
    food_perc = 1.0*food_pix/num_pix
    me_perc = 1.0*me_pix/num_pix

    snake_inds = np.nonzero(snake_layer)
    snake_inds = list(zip(snake_inds[0].tolist(),snake_inds[1].tolist()))

    food_inds = np.nonzero(food_layer)
    food_inds = list(zip(food_inds[0].tolist(),food_inds[1].tolist()))

    nearest_coord = self.get_closest_loc(food_inds)

    me_inds = np.nonzero(me_layer)
    me_inds = zip(me_inds[0].tolist(),me_inds[1].tolist())

    coord=[]
    for point in range(8):
      degree = point*(360//8)
      y = 30 * math.sin(math.radians(degree))
      x = 30 * math.cos(math.radians(degree))
      coord.append((270+x, 235+7))

    snake_dis = np.zeros(8)
    food_dis = np.zeros(8)
    danger_snake = np.zeros(8)

    for state in range(8):
      snake_dis[state] = min([self.d(i, coord[state]) for i in snake_inds]) if snake_inds else max_dis
      snake_dis[state] = snake_dis[state]*1.0/max_dis
      food_dis[state]  = min([self.d(i, coord[state]) for i in food_inds]) if food_inds else max_dis
      food_dis[state]  = 1.0*(max_dis - food_dis[state])/max_dis

    snake_perc, food_perc = self.get_perc_in_area(frame)
    danger_snake = self.snake_dis_in_area(snake_inds)

    min_snake = snake_dis*1.0/max_dis
    min_food  = 1.0*(max_dis - food_dis)/max_dis

    action = self.dodge_snake(snake_inds, food_inds)
    #features = np.array([me_perc , snake_perc, food_perc, min_snake, min_food, action[0], action[1]])

    features = np.array([snake_dis, food_dis, snake_perc, food_perc, danger_snake])

    return features[:, np.newaxis, np.newaxis]

  def d(self, ind, state):
    return abs(state[0]-ind[0]) + abs(state[1]-ind[1])

  def e(self, ind):
    return math.sqrt((270-ind[0])**2 + (235-ind[1])**2)

  def get_perc_in_area(self, frame):
    snake_perc = []
    food_perc = []
    x = ([271,520],[271,520],[187,352],[20,269],[20,269],[20,269],[187,352],[271,520])
    y = ([186,285],[85,234],[85,234],[85,234],[186,285],[236,385],[236,385],[236,385])

    for a in range(8):
      snake_layer = frame[x[a][0]:x[a][1], y[a][0]:y[a][1], 0]
      food_layer = frame[x[a][0]:x[a][1], y[a][0]:y[a][1], 2]

      num_pix = snake_layer.shape[0]*snake_layer.shape[1]

      snake_pix = np.count_nonzero(snake_layer)
      food_pix = np.count_nonzero(food_layer)

      snake_perc.append(1.0*snake_pix/num_pix)
      food_perc.append(1.0*food_pix/num_pix)

    return snake_perc, food_perc


  def snake_dis_in_area(self, snake_inds):
    snake_dis = np.zeros(8)

    x = ([271,520],[271,520],[187,352],[20,269],[20,269],[20,269],[187,352],[271,520])
    y = ([186,285],[85,234],[85,234],[85,234],[186,285],[236,385],[236,385],[236,385])
    action = list(range(8))
    done=[]
    
    for ind in snake_inds:
      for a in action:
        if ind[0] in range(x[a][0],x[a][1]) and ind[1] in range(y[a][0], y[a][1]):
          if self.e(ind) <100: 
            snake_dis[a] = 1
            done.append(a)
      action = list(set(action) - set(done))
      if len(action) == 0: break

    return snake_dis

  ### get the nearest item to the center(snake head)
  def get_closest_loc(self, foodlist):
    nearest_x = 1
    nearest_y = 2 
    min_val = 10000

    for ind in foodlist:

      val = abs(270-ind[0]) + abs(235-ind[1])
      ## get the food distance 
      if val < min_val:
        nearest_x = ind[0]
        nearest_y = ind[1]
        min_val = val

    return (nearest_x, nearest_y)

    # if no snake in the frame, look for the closest food
  def dodge_snake(self, snakelist, foodlist):
    if snakelist:
      nearest_coord_snake = self.get_closest_loc(snakelist)
      return (268*2 - nearest_coord_snake[0], 234*2 - nearest_coord_snake[1])
    else:
      return self.get_closest_loc(foodlist)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def create_slither_env(state_type):
  env = gym.make('internet.SlitherIO-v0')
  env = Vision(env)

  #Because logging is annoying
  #env = Logger(env)

  env = BlockingReset(env)

  env = CropScreen(env, 300, 500, 85, 20)
  #env = DiscreteToFixedKeysVNCActions(env, ['left', 'right', 'space', 'left space', 'right space'])
  env = EpisodeID(env)
  env = RenderWrapper(env, state_type)
  return env