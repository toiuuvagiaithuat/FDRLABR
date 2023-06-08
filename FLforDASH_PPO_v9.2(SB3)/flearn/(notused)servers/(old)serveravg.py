import torch
import os

from flearn.users.useravg import UserAVG
from flearn.users.userenv import Env
# from flearn.users.singlepath_env_gym import SinglepathEnvGym

from flearn.users.userenv import Env
# from flearn.servers.serverbase import Server
from flearn.users.DQN import DQN
from flearn.trainmodel.models import Model
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
from typing import Any, Dict
import config
import utils.bwlists
from utils.helper import list2csv, plot_reward



# Implementation for FedAvg Server

class FedAvg():
    def __init__(self):

        init_env = Env(0, config.env_config)
        self.model = Model(init_env.get_state_num(), init_env.get_action_num())

        self.num_glob_iters = config.system_config["num_glob_iters"]
        self.local_epochs = config.system_config["local_epochs"]
        self.users_per_round = config.system_config["users_per_round"]
        self.total_users = config.system_config["total_users"]
        self.times = config.system_config["times"]

        self.decay = config.client_config["decay"]
        self.batch_size = config.client_config["batch_size"]
        self.learning_rate = config.client_config["learning_rate"]
        self.max_experiences = config.client_config["max_experiences"]
        self.min_experiences = config.client_config["min_experiences"]
        self.ini_epsilon = config.client_config["ini_epsilon"]
        self.min_epsilon = config.client_config["min_epsilon"]
        self.gamma = config.client_config["gamma"]


        self.users = []
        for uid in range(self.total_users):
            # assign bw for the uid-th client
            if (config.env_config["is_synthetic_data"] == False):
                if uid in range(self.total_users // 2):
                    bw = utils.bwlists.get_fcc_train_data()
                else:
                    bw = utils.bwlists.get_lte_train_data()

                config.env_config['bw_list'] = bw

            env = Env(uid, config.env_config)

            self.rl_agent = DQN(env.get_state_num(), env.get_action_num())

            user = UserAVG(user_id=uid, rlagent=self.rl_agent, env=env)
            self.users.append(user)

        print("Number of users / total users:", self.users_per_round, " / " ,self.total_users)
        print("Finished creating FedAvg server.")

    # function calculates the weights of global model from averaging the weights from client models
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # if averaging over selected users
        ratio = 1 / len(self.selected_users)
        for user in self.selected_users:
            for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
                server_param.data += user_param.data.clone() * ratio

        # # if averaging over all users, however, the result is totally the same
        # ratio = 1 / len(self.users)
        # for user in self.users:
        #     for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
        #         server_param.data = server_param.data + user_param.data.clone() * ratio

    # broadcast weights of global model to users' models
    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model) # set client model's weights with global model's weights

    def select_users(self, round, users_per_round):
        if(users_per_round == len(self.users)):
            print("All users are selected")
            return self.users

        users_per_round = min(users_per_round, len(self.users))
        # fix the list of user consistent
        np.random.seed(round * (self.times + 1))
        return np.random.choice(self.users, users_per_round, replace=False) #, p=pk)


    def train(self):
        dir_name = config.log_config["dir_name"]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_name = config.log_config["data_filename"]
        model_name = config.log_config["model_name"]
        fig_name = config.log_config["fig_name"]

        self.reward_trace = []
        avg_user_reward = np.zeros(self.total_users)
        best_reward = -100
        for glob_iter in range(self.num_glob_iters):

            # send global model's parameter to clients (actually selected clients)
            self.send_parameters()

            self.selected_users = self.select_users(glob_iter, self.users_per_round)

            for user in self.selected_users:
                avg_user_reward[user.user_id] = user.train() #* user.train_samples

            avg_reward = sum(avg_user_reward)/self.total_users

            # for uid in range(len(self.users)):
            self.reward_trace = np.append(self.reward_trace, avg_reward)

            self.aggregate_parameters()

            print("Round number: ",glob_iter, " -- Average reward: ", avg_reward)

            list2csv(dir_name + "reward_system_"+file_name, [avg_reward])
            torch.save(self.model, os.path.join(dir_name, "model_" + model_name + ".pt"))

            # save the best model so far
            if best_reward < avg_reward:
                best_reward = avg_reward
                torch.save(self.model, os.path.join(dir_name, "best_model_" + model_name + ".pt"))
                torch.save(self.model.state_dict(), os.path.join(dir_name, "best_model_dict_" + model_name))


        plot_reward(dir_name + "fig_{}".format(fig_name)+ ".png", self.reward_trace)
