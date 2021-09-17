from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory

import cv2


class AgentFreezed(BaseModel):
  def __init__(self, config, environment, sess, model_type='ckptv2', prefix=None):
    super(AgentFreezed, self).__init__(config)
    self.sess = sess
    self.weight_dir = './checkpoints/BreakoutNoFrameskip-v0'
    self.model_type = model_type

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    self.input_name = 'prediction/s_t'
    self.output_name = 'prediction/ArgMax'

    self.load_dqn(prefix, model_type)

  def predict(self, s_t, test_ep=None):
    ep = test_ep
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      q = self.sess.run(self.output, feed_dict={self.input: [s_t]})
      action = q[0]
      # action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def load_dqn(self, prefix=None, model_type='ckptv2'):
    if prefix is not None:
      prefix = os.path.join(self.weight_dir, prefix)
    else:
      prefix = os.path.join(self.weight_dir, '-5710000')

    if model_type == 'ckptv1':
      ckpt = tf.train.get_checkpoint_state(self.weight_dir)
      assert ckpt is not None
      meta_file = prefix + '.meta'
      saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
      saver.restore(self.sess, prefix)
    elif model_type == 'ckptv2':
      meta_file = prefix + '.meta'
      saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
      input_checkpoint = meta_file[0: len(meta_file) - 5]
      saver.restore(self.sess, input_checkpoint)
    elif model_type == 'freezePb':
      pb_file = prefix + '.pb'
      pb = open(pb_file, 'rb')

      self.graph_def = tf.GraphDef()
      self.graph_def.ParseFromString(pb.read())
      tf.import_graph_def(self.graph_def, name='')

    # mark graph input + output
    input_tensor_name = self.input_name + ':0'
    self.input = self.sess.graph.get_tensor_by_name(input_tensor_name)
    output_tensor_name = self.output_name + ':0'
    self.output = self.sess.graph.get_tensor_by_name(output_tensor_name)

    return

  def save_dqn(self, saved_dir=None):
    if saved_dir is not None:
      pb_file = os.path.join(saved_dir, 'freezePb.pb')
    else:
      pb_file = 'freezePb.pb'

    output_node_name = self.output_name
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess=self.sess,
      input_graph_def=self.sess.graph_def,
      output_node_names=output_node_name.split(","))
    with tf.gfile.GFile(pb_file, "wb") as f:  # 保存模型
      f.write(output_graph_def.SerializeToString())  # 序列化输出

    return


  def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    # if not self.display:
    #   gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
    #   self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0

    self.env.env.reset()
    for idx in range(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      img_id = 24
      max_img_id = 24

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)
        # 4. display
        if not self.display:
          self.env.env.render(mode='human')
          np.testing.assert_almost_equal(test_history.get()[:, :, -1], screen, decimal=4)
          screen_saved = (screen * 255).astype(np.uint8)
          if img_id < max_img_id:
            cv2.imwrite('./images/BreakoutNoFrameskip-v0-{:02d}.jpg'.format(img_id), screen_saved)
            img_id += 1
          # time.sleep(0.01)
        else:
          img = self.env.env.render(mode='rgb_array')
          cv2.imshow('win', img)
          cv2.waitKey(0)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print("="*30)
      print(" [%d] Best reward : %d" % (best_idx, best_reward))
      print("="*30)

    self.env.env.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
