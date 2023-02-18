import numpy as np
import datetime
import torch
import gym
import os
import pathlib
import random
import matplotlib.pyplot as plt
from reward import Reward


class Human_Preference(object):
  def __init__(self, obs_size, action_size): 

    self.trajectories = []
    self.preferences = []
    self.batch_size = 7
    self.obs_size = obs_size
    self.action_size = action_size
    self.loss = []
    self.creat_model()


  def creat_model(self):
    self.model = Reward(input_dim=self.obs_size, num_actions=self.action_size)
    
  def train_model(self):
    self.model.train()
    if len(self.preferences) < 2:
      return 
    batch_size = min(len(self.preferences), self.batch_size)
    batch = random.sample(self.preferences, batch_size)
    loss_t = 0
    for i in range(len(batch)):
      query1, query2, human_pref = batch[i]
      loss_t =  self.model.train_step(query1, query2, human_pref)
      print('Batch: {}/{}, Loss: {}'.format(i+1, batch_size, loss_t))
    self.loss.append(loss_t)

  def predict(self, state):
    self.model.eval()
    out = torch.tensor([state])
    return self.model(out).detach().numpy()

  def add_pref(self, x1, x2, preference):
    self.preferences.append([x1, x2, preference])

  def add_trajectory(self, env,  trajectory):
    self.trajectories.append([env, trajectory])
   
  def ask_human_pref(self):
    if len(self.trajectories) < 2:  # not enough trajectories to compare
      return
    m, n = random.sample(range(len(self.trajectories)), 2)
    trajectories = [self.trajectories[m], self.trajectories[n]]
    traj_envs = []
    for i in range(2):
        traj_env, trajectory = trajectories[i]
        env = gym.make(traj_env)
        env.reset()
        for j in range(len(trajectory)):
          action = trajectory[j][2]
          env.step(action)
          env.render()
        traj_envs.append(env)
        env.close()
    preference = input("1, 2 or 3 if neutral: ")
    preference = int(preference) - 1

    sts = []
    for i in range(2):
        traj_env, trajectory = trajectories[i]
        st = []
        for j in range(len(trajectory)):
            st.append(trajectory[j][1])
        sts.append(st)
    self.add_pref(sts[0], sts[1], preference)

  def plot(self):
    x = np.arange(0, len(self.loss))
    y = np.asarray(self.loss)
    _, ax = plt.subplots()
    ax.plot(y)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per epoch')
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(pathlib.Path().absolute(), 'plots', 'hp' + datetime_str + ".png")
    plt.savefig(path)