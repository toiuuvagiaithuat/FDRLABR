import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
# from flearn.users.userbase import User
from flearn.users.userenv import Env
import pickle
import copy
import numpy as np
from utils.helper import list2csv, plot_reward, twodlist2csv
import config
# from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


from flearn.optimizers.fedoptimizer import *
# Implementation for FedAvg clients

# REWARD_TRACE_LENGTH = 6

class UserAVGPPO():
    def __init__(self, user_id, bw=None):
        info_keywords = ("reward_quality_norm", "reward_smooth_norm", "reward_rebuffering_norm")
        self.env = Monitor(Env(user_id, config.env_config, bw, istrain=True), info_keywords=info_keywords)
        self.rlagent = PPO("MultiInputPolicy", env = self.env, **config.PPO_params)

        self.user_id = user_id
        self.local_epochs = config.system_config['local_epochs']
        self.reward_trace = []

    def get_parameters(self):
        # for param in self.rlagent.q_net.parameters():
        #     param.detach()
        return self.rlagent.policy.parameters()

    # set parameter for client model to a new weight
    def set_parameters(self, new_model):
        for new_param, local_param in zip(new_model.parameters(), self.rlagent.policy.parameters()):
            local_param.data = new_param.data.clone()

    def train(self):
        dir_name = config.log_config["dir_name"]

        self.rlagent.learn(total_timesteps=59* self.local_epochs, reset_num_timesteps=False, tb_log_name=f"{dir_name}_logs")
        epoch_user_result = self.env.get_episode_rewards()[-self.local_epochs:]

        # list2csv(dir_name + "agent_reward_{}".format(self.user_id), epoch_user_result)

        return sum(epoch_user_result)/self.local_epochs


