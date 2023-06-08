
import torch.nn
from datetime import date, datetime
from torch import nn

system_config = {
    "num_glob_iters": 500,     # number of global iterations (rounds)
    "local_epochs": 20,         # number of epochs client ru before send its weights to server
    "test_period": 20,          # test for each 10 global iterations after exploration phase
    "users_per_round": 10,      # number of client chosen per round
    "total_users":100,         # number of clients
    "seed": 2
}


DQN_policy_kwargs = dict(
    net_arch=[64, 64],
    # activation_fn="Tanh"
)

DQN_params = dict(
        policy_kwargs=DQN_policy_kwargs,
        learning_rate=0.0005,
        buffer_size=1000,
        learning_starts=128,
        batch_size=128,
        # tau=args.dqn_tau,
        gamma=0.9,
        # train_freq=args.dqn_train_freq,
        # gradient_steps=args.dqn_grad_steps,
        target_update_interval=25,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        # max_grad_norm=args.dqn_max_grad_norm
)

A2C_policy_kwargs = dict(
        net_arch=[dict(pi=[64] * 3, vf=[64] * 2)],
#     activation_fn="Tanh"
)

A2C_params = dict(
        policy_kwargs=A2C_policy_kwargs,
        learning_rate=0.0005,
        gamma=0.9,
)

PPO_policy_kwargs = dict(
    net_arch=[dict(pi=[64] * 3, vf=[64] * 3)],
#     activation_fn="Tanh"
)
PPO_params = dict(
    policy_kwargs=PPO_policy_kwargs,
    n_steps=system_config["local_epochs"]*60,
    learning_rate=0.0001,
    batch_size=128,
    gamma=0.9,
    gae_lambda=0.95,
    clip_range=0.2,
)

env_config = {
    "log_qoe": True,
    "bw_list": None,
    "max_buffer": 20,
    "train_data": "realUpDown40", # "real", "fcc", "lte", realUpDown40
    "test_data": "realUpDown40",  # "real", "fcc", "lte", realUpDown40
}

log_config = {
    "dir_name": "./resultsTuned_meps01/" + str(env_config["train_data"]) + \
                "_c" + str(system_config["total_users"]) + \
                "_cpr" + str(system_config["users_per_round"]) + \
                "_globiter" + str(system_config["num_glob_iters"]) + \
                "_locepi" + str(system_config["local_epochs"]) +\
                "_seed" + str(system_config["seed"]) +  "/",

    "model_name": "model_" + str(env_config["train_data"]) + \
                "_c" + str(system_config["total_users"]) + \
                "_cpr" + str(system_config["users_per_round"]) + \
                "_globiter" + str(system_config["num_glob_iters"]) + \
                "_locepi" + str(system_config["local_epochs"]) +\
                "_seed" + str(system_config["seed"]),
    "data_filename": "reward_" + str(env_config["train_data"]) + \
                "_c" + str(system_config["total_users"]) + \
                "_cpr" + str(system_config["users_per_round"]) + \
                "_globiter" + str(system_config["num_glob_iters"]) + \
                "_locepi" + str(system_config["local_epochs"]) +\
                "_seed" + str(system_config["seed"]),
    "fig_name": "fig" + str(env_config["train_data"]) + \
                "_c" + str(system_config["total_users"]) + \
                "_cpr" + str(system_config["users_per_round"]) + \
                "_globi" + str(system_config["num_glob_iters"]) + \
                "_loci" + str(system_config["local_epochs"]) +\
                "_seed" + str(system_config["seed"])
    }

if __name__ == '__main__':

    # current date and time
    now = datetime.now()
