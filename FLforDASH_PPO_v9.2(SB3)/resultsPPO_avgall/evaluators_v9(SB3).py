from flearn.users.userenv import Env
import numpy as np
# from utils import DownloadPath
# from feature_extractor import PensieveFeatureExtractor
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN, A2C
import torch
import wandb
# from wandb.integration.sb3 import WandbCallback
import time
import datetime
import config
import utils.bwlists
# from flearn.users.DQN import DQN
# from stable_baselines.common.evaluation import evaluate_policy

# NUM_STEP_PER_EP = 59
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
from utils.helper import list2csv, plot_reward, twodlist2csv
from flearn.users.useravg import UserAVG
import config

class BaseEvaluator:
    """
    Base evaluator for inheritance
    """

    def __init__(self, data):
        self.data = data
        # config.client_config["train"] = False
        self.istrain = False
        self.VIDEO_CHUNK_LEN = 4
        self.M_IN_K =1000

        if (self.data=='fcc'):
            self.bw = utils.bwlists.get_fcc_test_data()
        elif (self.data=='lte'):
            self.bw = utils.bwlists.get_lte_test_data()
        elif (self.data=='real'):
            self.bw = utils.bwlists.get_real_test_data()
        elif (self.data == 'realUpDown40'):
            self.bw = utils.bwlists.get_real_test800_data_updown40()
        else:
            print("Error input!")

        self.env = Env(0, config.env_config, self.bw, istrain=False)

    def predict(self, *args):
        pass

    def evaluate(self, *args):
        pass


class ConstantEvaluator(BaseEvaluator):
    """
    An dummy evaluator using constant prediction.

    Always return the max quality.
    """

    def predict(self):
        # Always return max quality
        return 3

    def evaluate(self, file_name):
        test_result = []
        for eps in range(len(self.bw)):
            self.env.reset()
            done = False
            while not done:
                predicted_action = self.predict()
                state, reward, done, info = self.env.step(predicted_action)
            epi_utility = info["reward_quality_norm"]
            epi_switch_penalty = info["reward_smooth_norm"]
            epi_rebuffering_penalty = info["reward_rebuffering_norm"]
            epi_reward = info["sum_reward"]

            # print(eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty)
            test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

        twodlist2csv(file_name, test_result)


class SmoothEvaluator(BaseEvaluator):
    """
    Smooth Throughput evaluator.
    Use mean (or harmonic mean) of window_size (typically 3) previous network speed to predict the next quality,
        such that it does not exceed the bitrates of the next segment.
    """

    def predict(self, video_list, prev_network_speed, rule="harmonic_mean", window_size=3, scale = 0.9):
        """
        Predict the next qualities based on previous network speed.
        :param video_list: np.array of shape (7, CHUNK_TIL_VIDE_END), row is the quality level and column is its sizes.
            It is self.env.video_list
        :param prev_network_speed: np.array of shape (HISTORY_SIZE,), previous network speed of a path
        :param rule: string, either mean or harmonic_mean, indicates which rule to calculate quality
        :param window_size: int, default 3, how many previous network speed values to use.
        :return:
        """
        down_id = self.env.pick_next_segment()
        segment_bitrates = video_list[:, down_id] * 8 / self.env.VIDEO_CHUNK_LEN
        prev_network_speed = prev_network_speed[:window_size].copy()
        prev_network_speed = prev_network_speed[prev_network_speed > 0].copy()  # Eliminate zeros (values that are unfilled)
        if len(prev_network_speed) == 0:
            return 0
        if rule == "mean":
            predicted_quality = sum(prev_network_speed) / len(prev_network_speed)
        elif rule == "harmonic_mean":
            predicted_quality = len(prev_network_speed) / np.sum(1 / prev_network_speed)
        else:
            raise AssertionError("Rule is not mean or harmonic_mean")

        picked_quality = 0
        if segment_bitrates[picked_quality] >= predicted_quality * scale * 1000 ** 2:
            return picked_quality
        while segment_bitrates[picked_quality] < predicted_quality * scale * 1000 ** 2:
            picked_quality += 1
            if picked_quality == self.env.get_action_num():
                break
        return picked_quality - 1

    def evaluate(self, file_name):
        test_result = []

        for eps in range(len(self.bw)):
            # cur_path = DownloadPath.PATH1
            state = self.env.reset()

            done=False
            while not done:
                predicted_action = self.predict(self.env.video_list, state["network_speed"])
                state, reward, done, info = self.env.step(predicted_action)

            epi_utility = info["reward_quality_norm"]
            epi_switch_penalty = info["reward_smooth_norm"]
            epi_rebuffering_penalty = info["reward_rebuffering_norm"]
            epi_reward = info["sum_reward"]

            # print(eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty)
            test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

        twodlist2csv(file_name, test_result)


class RandomEvaluator(BaseEvaluator):
    def predict(self):
        return np.random.choice(np.arange(7))

    def evaluate(self, file_name):
        test_result = []

        for eps in range(len(self.bw)):
            self.env.reset()
            done=False
            while not done:
                predicted_action = self.predict()
                state, reward, done, info = self.env.step(predicted_action)

            epi_utility = info["reward_quality_norm"]
            epi_switch_penalty = info["reward_smooth_norm"]
            epi_rebuffering_penalty = info["reward_rebuffering_norm"]
            epi_reward = info["sum_reward"]

            # print(eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty)
            test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])
        twodlist2csv(file_name, test_result)

class BolaEvaluator(BaseEvaluator):
    def predict(self, playtime_from_begin, playtime_to_end, prev_quality, cur_buffer, prev_bw):
        """
        Implement BOLA Algorithm

        Args:
            Q(t_k): the buffer level at the start of the slot k (in seconds).\n
            Q_max: buffer therhhold (in seconds).\n
            Q^D_max: dynamic buffer level. (in seconds).\n
            Q: current buffer level (in second).\n
            S_m: the size of any segment encoded at bitrate index m  (in bits).\n
            v_m: ln(S_m/S_1) utility function.\n
            V>0: 0.93 to allow a tradeoff between the buffer size and the performance objectives.\n
            V_D: Dynamic V which corresponds to a dynamic buffer size Q^D_max.\n
            m*: The index that maximizes the ratio among all m for which this ratio is positive.\n
            m*[n]: Size of segment n at bitrate index m*.\n
            m*[n-1]: Size of segment n-1 at bitrate index m*.\n
            p: video segment (in second)


        :param (,7) next_seg: n segment sizes need choice one for download (in bps)
        :param float playtime_from_begin:
        :param float playtime_to_end:
        :param float prev_quality: bandwidth measured when downloading segment n-1
        :param float cur_buffer: buffer level at the time start to download segment n
        :param float prev_bw: estimated bandwidth (in bps) on the considering path
        :param (,7) S_m: the size of any segment encoded at bitrate index m  (in bits)
        :return: the quality index for download
        """

        p = self.VIDEO_CHUNK_LEN
        Q_MAX = self.env.BUFFER_THRESHOLD / p
        S_m = np.array(self.env.VIDEO_BIT_RATE) * 4.0 * self.M_IN_K

        v = np.log(S_m / S_m[0])
        gamma = 5.0 / p  # CHANGE from 5.0/p

        t = min(playtime_from_begin, playtime_to_end)
        t_prime = max(t / 2.0, 3 * p)
        Q_D_max = min(Q_MAX, t_prime / p)
        V_D = (Q_D_max - 1) / (v[-1] + gamma * p)  # v[-1] is v_M

        m_star = 0
        score = None
        for q in range(len(S_m)):
            s = ((V_D * (v[q] + p * gamma) - cur_buffer / p) / S_m[q])
            if score is None or s > score:
                m_star = q
                score = s

        if m_star > prev_quality:
            r = prev_bw * 10 ** 6  # Calculate in bits
            m_prime = np.where(S_m / p <= max(r, S_m[0] / p))[0][-1]
            if m_prime >= m_star:
                m_prime = m_star
            elif m_prime < prev_quality:
                m_prime = prev_quality

            else:
                m_prime = m_prime + 1
            m_star = m_prime
        return m_star

    def evaluate(self, file_name):
        test_result = []

        for eps in range(len(self.bw)):
            state = self.env.reset()
            playtime_from_begin = 0
            prev_network_speed = 0
            done = False
            while not done:
                down_id = self.env.pick_next_segment()
                playtime_to_end = self.env.CHUNK_TIL_VIDEO_END - self.env.play_id * self.env.VIDEO_CHUNK_LEN
                prev_quality = self.env.download_segment[down_id - 1]
                cur_buffer = self.env.buffer_size_trace
                predicted_action = self.predict(playtime_from_begin, playtime_to_end, prev_quality,
                                                cur_buffer, prev_network_speed)
                state, reward, done, info = self.env.step(predicted_action)

                prev_network_speed = state["network_speed"][0]
                playtime_from_begin = self.env.play_id * self.VIDEO_CHUNK_LEN \
                                      + self.env.rebuffer_time

            epi_utility = info["reward_quality_norm"]
            epi_switch_penalty = info["reward_smooth_norm"]
            epi_rebuffering_penalty = info["reward_rebuffering_norm"]
            epi_reward = info["sum_reward"]

            test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

        twodlist2csv(file_name, test_result)

class DQNEvaluator(BaseEvaluator):
    def evaluate(self, file_name, model_file):
        test_agent = DQN.load(model_file, env = self.env)

        test_result = []
        for eps in range(len(self.bw)):
            observations = self.env.reset()

            done = False
            while not done:
                predicted_action, _ = test_agent.predict(observations, deterministic=True)
                observations, reward, done, info = self.env.step(predicted_action)

            epi_utility = info["reward_quality_norm"]
            epi_switch_penalty = info["reward_smooth_norm"]
            epi_rebuffering_penalty = info["reward_rebuffering_norm"]
            epi_reward = info["sum_reward"]
            test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

        twodlist2csv(file_name, test_result)


class PPOEvaluator(BaseEvaluator):
    def evaluate(self, file_name, model_file):
        test_agent = PPO.load(model_file, env = self.env)

        test_result = []
        for eps in range(len(self.bw)):
            observations = self.env.reset()

            done = False
            while not done:
                predicted_action, _ = test_agent.predict(observations, deterministic=True)
                observations, reward, done, info = self.env.step(predicted_action)

            epi_utility = info["reward_quality_norm"]
            epi_switch_penalty = info["reward_smooth_norm"]
            epi_rebuffering_penalty = info["reward_rebuffering_norm"]
            epi_reward = info["sum_reward"]
            test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

        twodlist2csv(file_name, test_result)

if __name__ == '__main__':
    data = 'realUpDown40'

    eval=ConstantEvaluator(data)
    eval.evaluate('test_constant3_B' + str(config.env_config["max_buffer"]) + data)

    # eval = SmoothEvaluator(data)
    # eval.evaluate('test_smooth09_B' + str(config.env_config["max_buffer"]) + data)
    # # #
    # eval = BolaEvaluator(data)
    # eval.evaluate('test_bola_B'  + str(config.env_config["max_buffer"]) + data)
    #
    # eval = RandomEvaluator(data)
    # eval.evaluate('test_random_B' + str(config.env_config["max_buffer"]) + data)
    #
    # model_file = "C:/Users/DELL/Documents/FDRLABR_v9.2/PPO/PPO_local20/realUpDown40_c100_cpr10_globiter500_locepi20_seed4/bestmodel.zip"
    # eval = DQNEvaluator(data)
    # eval = PPOEvaluator(data)

    # eval.evaluate(model_file, model_file)
