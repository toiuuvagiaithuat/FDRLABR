# from flearn.servers.serveravg import FedAvg
from flearn.servers.serveravgA2C import FedAvgA2C
import utils.utils as helper
import config


if __name__ == "__main__":
    helper.set_global_seed(config.system_config["seed"])

    print("=" * 80)
    print("FL PARAMETERS:")
    print("Subset of users: {}".format(config.system_config['users_per_round']))
    print("Total users : {}".format(config.system_config['total_users']))
    print("Number of local rounds       : {}".format(config.system_config['local_epochs']))
    print("Number of global rounds       : {}".format(config.system_config['num_glob_iters']))

    # print("RL AGENT PARAMETERS:")
    # print("Fraction: {}".format(config.DQN_params["exploration_fraction"]))
    # print("Batch size: {}".format(config.DQN_params['batch_size']))
    # print("Learning rate       : {}".format(config.DQN_params['learning_rate']))
    print("SEED       : {}".format(config.system_config["seed"]))
    print("=" * 80)

    print("Data Train       : {}".format(config.env_config["train_data"]))
    print("Data Test       : {}".format(config.env_config["test_data"]))
    print("=" * 80)

    # train
    # server = FedAvg()
    server = FedAvgA2C()
    server.train()

