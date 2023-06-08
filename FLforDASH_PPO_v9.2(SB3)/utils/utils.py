from enum import Enum
import requests
import numpy as np
import pickle
import pandas as pd
import random
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
import time
import json


# class NetworkProfileReader:
#     def __init__(self, profile_name="profile1", file_path="dash_network_profiles.json", scale=1000):
#         """
#          Read the Network Profile json file.
#
#         :param profile_name: profile to load. Valid name is "profile" + x where x is in range [1, 12].
#             For example, "profile1", "profile8", "profile12".
#         :param file_path: path to the json file
#         :param scale: to scale the bw to bps, since DASH uses kbps
#         """
#         assert profile_name in ["profile" + str(x) for x in range(1, 13)]
#         with open(file_path) as json_file:
#             data = json.load(json_file)
#             self.bw = np.array(data["data"][profile_name]["bw"]) * scale
#             self.delay = np.array(data["data"][profile_name]["delay"])
#             self.packet_loss = np.array(data["data"][profile_name]["packet_loss"])
#             self.pattern = data["data"][profile_name]["pattern"]
#             json_file.close()
#
#
# class LTEProfileReader:
#     def __init__(self, vehicle="bus", profile_name="profile1", file_path="LTE_bitrate.json", scale=1):
#         """
#          Read the Network Profile json file.
#
#         :param profile_name: profile to load. Valid name is "profile" + x where x is in range [1, 12].
#             For example, "profile1", "profile8", "profile12".
#         :param file_path: path to the json file
#         :param scale: to scale the bw to bps, since DASH uses kbps
#         """
#         with open(file_path, "r") as file:
#             data = json.load(file)
#             vehicles = ['bus', 'static', 'pedestrian', 'car', 'train']
#             assert vehicle in vehicles
#             available = data["data"][vehicle].keys()
#             assert profile_name in [x for x in data["data"][vehicle].keys()], \
#                 f"Profile name not available. Your input: {profile_name}, available: {available}"
#             self.bw = data["data"][vehicle][profile_name] * scale
#             file.close()
#

class Event(Enum):
    DOWN = 0
    DOWN_FINISH = 1
    PLAY = 2
    PLAY_FINISH = 3
    SLEEP_FINISH = 4
    FREEZE_FINISH = 5


class VideoListCollector:
    def __init__(self,
                 base_link="http://ftp.itec.aau.at/datasets/mmsys12/ElephantsDream/ed_4s/",
                 form="ed_4sec_{0}kbit/ed_4sec{1}.m4s",
                 save_dir="data/video_list_bunny8k",
                 available_bitrate=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 900, 1200, 1500,
                                    2000, 2500, 3000, 4000, 5000, 6000, 8000]
                 ):
        self.available_bitrate = available_bitrate
        self.base_link = base_link
        self.form = form
        self.save_dir = save_dir

    def seperate_trace(self):
        self.segment_trace = {}
        for x in self.available_bitrate:
            seg_size = []
            seg_num = 1
            while True:
                link = self.base_link + self.form.format(str(x), str(seg_num))
                r = requests.head(link)
                try:
                    seg_size.append(r.headers['Content-Length'])
                    seg_num += 1
                except:
                    break
            if seg_num > 2:
                self.segment_trace['{0}'.format(str(x))] = seg_size
                print('collect for {}kbit trace completed'.format(str(x)))
            else:
                print('video has no bitrate level {0}kbit'.format(str(x)))

    def get_trace_matrix(self, bitrate_list):
        self._load()
        return_matrix = []
        for bitrate in bitrate_list:
            try:
                return_matrix.append(self.segment_trace[str(bitrate)])
            except:
                print('trace for bitrate {0}kbit does not exist'.format(str(bitrate)))
        return np.asarray(return_matrix, dtype=np.float32)

    def save(self):
        pickle.dump(self.segment_trace, open(self.save_dir, 'wb'))

    def _load(self):
        self.segment_trace = pickle.load(open(self.save_dir, 'rb'))


def result_plotter(file, name, col="r"):
    if col == "r":
        df = pd.read_csv(file, skiprows=[0])
    else:
        df = pd.read_csv(file)
    plt.figure(figsize=(10, 15))
    sns.lineplot(y=np.cumsum(df[col].values), x=list(range(len(df))))
    plt.title(name)
    plt.show()
    time.sleep(5)
    plt.close()


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # v = VideoListCollector()
    # v.seperate_trace()
    # v.save()

    result_plotter("results/testDQN.monitor.csv", name="DQN")

