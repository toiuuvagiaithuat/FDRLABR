import numpy as np
import torch
import torch.optim as optim
import config
from flearn.trainmodel.models import Model

class DQN:
    def __init__(self, num_states, num_actions):

        self.batch_size = config.client_config['batch_size']
        self.lr = config.client_config["learning_rate"]
        self.gamma = config.client_config["gamma"]
        self.max_experiences = config.client_config["max_experiences"]
        self.min_experiences = config.client_config["min_experiences"]
        self.ini_epsilon = config.client_config["ini_epsilon"]
        self.min_epsilon = config.client_config["min_epsilon"]
        self.decay = config.client_config["decay"]
        self.l2_weight = config.client_config["l2_weight"]

        # self.istrain = config.client_config["train"]

        self.num_actions = num_actions
        self.num_states = num_states
        self.model = Model(self.num_states, self.num_actions)
        
        if self.l2_weight != 0:
            self.model.optimizer = optim.Adam(self.model.parameters(), \
                                          lr=self.lr, \
                                          weight_decay=self.l2_weight)
        else:
            self.model.optimizer = optim.Adam(self.model.parameters(), self.lr)

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}


    def update(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0

        self.model.optimizer.zero_grad()

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)

        states = torch.tensor(np.array([self.experience['s'][i] for i in ids]), dtype=torch.float32)
        actions = torch.tensor(np.array([self.experience['a'][i] for i in ids]), dtype=torch.long)
        rewards = torch.tensor(np.array([self.experience['r'][i] for i in ids]), dtype=torch.float32)
        states_next = torch.tensor(np.array([self.experience['s2'][i] for i in ids]), dtype=torch.float32)
        dones = torch.tensor(np.array([self.experience['done'][i] for i in ids]), dtype=torch.bool)

        value_next = TargetNet.model.predict(states_next).max(dim=1)[0]

        actual_values = torch.where(dones, rewards, rewards+self.gamma*value_next)

        indices = torch.arange(self.batch_size, dtype=torch.long)

        selected_action_values = self.model.predict(states)[indices, actions]
        loss = self.model.loss(actual_values, selected_action_values).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

    def get_action(self, state, epsilon):
        if ((np.random.random() < epsilon)):
            return np.random.choice(self.num_actions)
        else:
            state_tensor = torch.tensor(np.array([state]),dtype = torch.float).to(self.model.device)
            actions = self.model.predict(state_tensor)
            action = torch.argmax(actions).item()
            return action

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
                # self.experience[key] = self.experience[key][1:]
        for key, value in exp.items():
            self.experience[key].append(value)
            # self.experience[key] = np.append(self.experience[key], value, axis=1)

    # def copy_weights(self, TrainNet):
    #     self.model.load_state_dict(self.TrainNet.state_dict())

    
    # def save_model(self, path):
    #     torch.save(self.model.state_dict(), path)

    # def load_model(self,path):
    #     self.model.load_state_dict(torch.load(path))

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load(path)
