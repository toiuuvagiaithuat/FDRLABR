import torch
import os

# from flearn.users.useravg import UserAVG
from flearn.users.useravgA2C import UserAVGA2C

from flearn.users.userenv import Env
from stable_baselines3 import PPO, DQN, A2C

import numpy as np
import copy
from typing import Any, Dict
import config
import utils.bwlists
from utils.helper import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
# import wandb
# from wandb.integration.sb3 import WandbCallback
# import time
# import datetime
# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.env_util import make_vec_env


# Implementation for FedAvg Server

class FedAvgPPO():
    def __init__(self):
        init_env = Monitor(Env(0, config.env_config))
        ppo = PPO("MultiInputPolicy", env=init_env, **config.PPO_params)
        self.model = ppo.policy # global model

        self.num_glob_iters = config.system_config["num_glob_iters"]
        self.local_epochs = config.system_config["local_epochs"]
        self.users_per_round = config.system_config["users_per_round"]
        self.total_users = config.system_config["total_users"]
        self.seed = config.system_config["seed"]

        self.users = []
        for uid in range(self.total_users):
            # assign bw for the uid-th client
            if (config.env_config["train_data"] == "real"):
                if uid in range(self.total_users // 2):
                    bw = utils.bwlists.get_fcc_train_data()
                else:
                    bw = utils.bwlists.get_fcc_train_data()
            elif (config.env_config["train_data"] == "fcc"):
                bw = utils.bwlists.get_fcc_train_data()
            elif (config.env_config["train_data"] == "lte"):
                bw = utils.bwlists.get_lte_train_data()
            elif (config.env_config["train_data"] == "realUpDown40"):
                if uid in range(self.total_users // 4):
                    bw = utils.bwlists.get_fcc_up40()
                elif uid in range(self.total_users // 4, 2*self.total_users // 4):
                    bw = utils.bwlists.get_fcc_low40()
                elif uid in range(2*self.total_users // 4, 3*self.total_users // 4):
                    bw = utils.bwlists.get_lte_up40()
                else:
                    bw = utils.bwlists.get_lte_low40()  # generate in the environment for each episode

            user = UserAVGPPO(user_id=uid, bw = bw)
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

        np.random.seed(round*(self.seed+1))
        return np.random.choice(self.users, users_per_round, replace=False) #, p=pk)


    def train(self):
        dir_name = config.log_config["dir_name"]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.reward_trace = []
        avg_user_reward = np.zeros(self.total_users)
        best_reward = -100

        for glob_iter in range(self.num_glob_iters):
            # send global model's parameter to clients (actually selected clients)
            self.send_parameters()

            self.selected_users = self.select_users(glob_iter, self.users_per_round)

            for user in self.selected_users:
                avg_user_reward[user.user_id] = user.train()#(n_eval_episodes = self.local_epochs) #* user.train_samples

            avg_reward = sum(avg_user_reward)/self.total_users

            # for uid in range(len(self.users)):
            self.reward_trace = np.append(self.reward_trace, avg_reward)

            self.aggregate_parameters()

            print("Round number: ", glob_iter, " -- Average reward: ", avg_reward)

            list2csv(dir_name + "reward_system", [avg_reward])

            if (((glob_iter+1)%config.system_config["test_period"] == 0) and (glob_iter >= (self.num_glob_iters)*2/3)):
                # torch.save(self.model, os.path.join(dir_name, config.log_config["model_name"] + '_' + str(glob_iter+1) + ".pt"))
                self.users[0].rlagent.save(f"{dir_name}_{str(glob_iter+1)}")

                for u in self.users:
                    list2newcsv_col(dir_name + "agent_reward_{}".format(u.user_id), u.env.get_episode_rewards())

                # validate
                if (config.env_config["test_data"] == 'fcc'):
                    bw = utils.bwlists.get_fcc_test_data()
                elif (config.env_config["test_data"] == 'lte'):
                    bw = utils.bwlists.get_lte_test_data()
                elif (config.env_config["test_data"] == 'real'):
                    bw = utils.bwlists.get_real_test_data()
                elif (config.env_config["test_data"] == 'realUpDown40'):
                    bw_validate = utils.bwlists.get_real_valid_data_updown40()
                    bw_test = utils.bwlists.get_real_test800_data_updown40()
                else:
                    print("Error input!")

                validate_agent = self.users[0].rlagent
                info_keywords = ("reward_quality_norm", "reward_smooth_norm", "reward_rebuffering_norm")
                env_valid = Monitor(env=Env(0, config.env_config, bw_validate, istrain=False), info_keywords=info_keywords)
                # env_valid = Env(0, config.env_config, bw_validate)
                validate_agent.set_env(env=env_valid)

                result_validate, _ = evaluate_policy(validate_agent, env_valid, n_eval_episodes=len(bw_validate),
                                                return_episode_rewards=True, deterministic=True)
                # list2newcsv_col(dir_name + "validate_" + config.env_config["test_data"] + '_' + str(glob_iter+1), result_validate)

                avg_validate = sum(result_validate)/len(result_validate)
                list2csv(dir_name + "validate_breakpoints", [glob_iter+1, avg_validate])

            # test
            if (glob_iter == self.num_glob_iters - 1):
                test_agent = self.users[0].rlagent
                info_keywords = ("reward_quality_norm", "reward_smooth_norm", "reward_rebuffering_norm")
                env_test = Monitor(env=Env(0, config.env_config, bw_test, istrain=False), info_keywords=info_keywords)
                test_agent.set_env(env=env_test)

                test_result = []
                for eps in range(len(bw_test)):
                    observations = env_test.reset()
                    done = False
                    while not done:
                        predicted_action, states = test_agent.predict(observations, deterministic=True)
                        observations, reward, done, info = env_test.step(predicted_action)

                    epi_utility = info["reward_quality_norm"]
                    epi_switch_penalty = info["reward_smooth_norm"]
                    epi_rebuffering_penalty = info["reward_rebuffering_norm"]
                    epi_reward = info["sum_reward"]

                    # print(eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty)
                    test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

                twodlist2csv(dir_name + "test_" + str(glob_iter+1), test_result)


            # save the best model
            if best_reward < avg_reward:
                best_reward = avg_reward
                self.users[0].rlagent.save(f"{dir_name}bestmodel")

                # torch.save(self.model, os.path.join(dir_name, "best_" + config.log_config["model_name"] + ".pt"))


        plot_reward(dir_name + config.log_config["fig_name"] + ".png", self.reward_trace)
