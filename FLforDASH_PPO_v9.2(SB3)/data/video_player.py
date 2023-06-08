import numpy as np
import math
from get_down_size import video_list_collector

VIDEO_CHUNCK_LEN = 4                            # sec, every time add this amount to buffer
BITRATE_LEVELS = 7
TOTAL_VIDEO_CHUNCK = 60                          # number of video segment per training section
BUFFER_THRESH = 30.0                            # sec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 0.5                   # sec
NETWORK_SEGMENT = 1                             # sec

class VideoPlayer():

    def __init__(self, bitrate_list, video_list):

        self.bitrate_list = bitrate_list
    #     np.array([500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,
    # 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000,]) 
        self.video_list = video_list

        self.buffer_thresh = BUFFER_THRESH
        self.reset()

    def reset(self, mode = 'train'):
        if mode == 'test':
            self.net_seg_iterator = 0    
            self.video_ilterator = 0
        else:
            self.net_seg_iterator = np.random.randint(0, len(self.bitrate_list)-1) 
            self.video_ilterator = np.random.randint(0, len(self.video_list[0])-1)

        self.video_seg_download = 0
        self.seg_time_stamp = 0
        self.buffer = 0
        return self.video_list[:,self.video_ilterator] 

    def download(self, bitrate_level):
        assert bitrate_level >=0
        assert bitrate_level < BITRATE_LEVELS

        delay = 0
        sleep_time = 0
        freeze_start = self.net_seg_iterator + self.seg_time_stamp + self.buffer

        # sleep if buffer is full
        while self.buffer + VIDEO_CHUNCK_LEN > self.buffer_thresh: 
            self.buffer -= DRAIN_BUFFER_SLEEP_TIME
            delay += DRAIN_BUFFER_SLEEP_TIME                                    #in second
            self.seg_time_stamp += DRAIN_BUFFER_SLEEP_TIME                      #in second
            sleep_time += DRAIN_BUFFER_SLEEP_TIME                               #in second

        # get network iterator to position after sleeping and download last segment
        pass_seg = math.floor(self.seg_time_stamp / NETWORK_SEGMENT)
        self.net_seg_iterator += pass_seg 
        self.seg_time_stamp -= pass_seg * NETWORK_SEGMENT

        #download video segment in bytes
        segment = float(self.video_list[bitrate_level][self.video_ilterator])          
        return_segment = segment

        while True:                                                                 #download segment process finish after a full video segment is downloaded
            self.net_seg_iterator = self.net_seg_iterator % len(self.bitrate_list)  #loop back to begin if finished
            network = self.bitrate_list[self.net_seg_iterator]                      #network DL_bitrate in bps
            max_throughput = network * (NETWORK_SEGMENT - self.seg_time_stamp)      #maximum possible throughput in bytes

            if max_throughput > segment:                                        #finish download in network segment
                self.seg_time_stamp += segment / network                        #used time in network segment in second
                delay += segment / network                                      #delay from begin in second
                break
            else:                                                               
                delay += NETWORK_SEGMENT - self.seg_time_stamp                  #delay from begin in second
                self.seg_time_stamp = 0                                         #used time of next network segment is 0s
                segment -= max_throughput                                       #remain undownloaded part of video segment
                self.net_seg_iterator += 1                                      #move to next network segment

        rebuf = max(0,delay - self.buffer)                                      #delay time in video player in second      
        self.buffer = max(0, self.buffer - delay) + VIDEO_CHUNCK_LEN            #remain buffer in second
        return_buffer = self.buffer                                             #in second
        self.video_seg_download += 1                                            #total downloaded segment
        remain = TOTAL_VIDEO_CHUNCK - self.video_seg_download                   #remain video segment in episode
        self.video_ilterator = (self.video_ilterator + 1) % len(self.video_list[0]) #loop back if finished video
        terminate = False

        if self.video_seg_download >= TOTAL_VIDEO_CHUNCK:                       #if terminate reset
            terminate = True
            self.reset()
        next_segments = self.video_list[:,self.video_ilterator]                 #get sizes of next download options
        return delay, sleep_time, return_buffer, rebuf, return_segment, next_segments, terminate, remain 

    def set_buffer(self, size):
        if size != None:
            self.buffer_thresh = size

    def get_buffer_thresh(self):
        return self.buffer_thresh