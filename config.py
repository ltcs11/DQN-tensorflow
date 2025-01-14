class AgentConfig(object):
  scale = 10000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NHWC'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 0.00025
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 12
  train_frequency = 6
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

class EnvironmentConfig(object):
  env_name = 'BreakoutNoFrameskip-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

# class EnvironmentConfig_Breakout(object):
#   env_name = 'BreakoutNoFrameskip-v0'
#
#   screen_width  = 84
#   screen_height = 84
#   max_reward = 1.
#   min_reward = -1.
#
# class EnvironmentConfig_SpaceInvaders(object):
#   env_name = 'SpaceInvadersNoFrameskip-v4'
#
#   screen_width  = 84
#   screen_height = 84
#   max_reward = 1.
#   min_reward = -1.
#
# class EnvironmentConfig_Pong(object):
#   env_name = 'PongNoFrameskip-v4'
#
#   screen_width  = 84
#   screen_height = 84
#   max_reward = 1.
#   min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 2

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    raise NotImplementedError

  for k in FLAGS.__dict__['__wrapped']:
      if k == 'use_gpu':
          if not FLAGS.__getattr__(k):
              config.cnn_format = 'NHWC'
          else:
              config.cnn_format = 'NHWC'

  if hasattr(config, k):
      setattr(config, k, FLAGS.__getattr__(k))

  return config
