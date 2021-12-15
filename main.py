from __future__ import print_function
import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.agent_freezed import AgentFreezed
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment (default if not in config.py)
flags.DEFINE_string('env_name', 'BreakoutNoFrameskip-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', False, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

# Self
flags.DEFINE_boolean('is_load_only', True, 'Whether to load agent directly from ckpt only')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  # gpu_options = tf.GPUOptions(
  #     per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  gpu_options = tf.GPUOptions(allow_growth=True)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    else:
      env = GymEnvironment(config)

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    if not FLAGS.is_load_only:
      agent = Agent(config, env, sess)
      if FLAGS.is_train:
        agent.train()
      else:
        agent.play()
    else:
      load_model_type = 'SavedModel'
      # save_model_type = 'ckptv1'
      print('Using agent_freezed! Some items in config are invalid')
      agent = AgentFreezed(config, env, sess, prefix='saved_model', model_type=load_model_type)
      # agent.save_dqn(saved_dir=agent.weight_dir, model_type=save_model_type)
      agent.play()


if __name__ == '__main__':
  tf.app.run()
