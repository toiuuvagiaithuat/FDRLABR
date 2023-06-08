import requests
import numpy as np
import pickle


class video_list_collector:
    def __init__(self,
                base_link = "http://ftp.itec.aau.at/datasets/mmsys12/ElephantsDream/ed_4s/",
                form = "ed_4sec_{0}kbit/ed_4sec{1}.m4s",
                save_dir = "video_list_new"):
        self.base_link = base_link
        self.form = form
        self.save_dir = save_dir
        self.__load()

    # def seperate_trace(self,available_bitrate = [
    #         50,100,150,200,250,300,400,500,600,700,900,1200,1500,2000,2500,3000, 4000, 5000, 6000, 8000
    #                 ]):
    #
    def seperate_trace(self, available_bitrate = [300, 700, 1200, 1500, 3000, 4000, 5000, 6000, 8000]):
        self.available_bitrate = available_bitrate
        self.segment_trace = {}
        for x in available_bitrate:
            seg_size = []
            seg_num = 1
            while(True):
                link = self.base_link + self.form.format(str(x),str(seg_num))
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
                print('video have no bitrate level {0}kbit'.format(str(x)))

    def get_trace_matrix(self, bitrate_list):
        return_matrix = []
        for bitrate in bitrate_list:
            try:
                return_matrix.append(self.segment_trace[str(bitrate)])
            except:
                print('trace for bitrate {0}kbit does not exist'.format(str(bitrate)))
        
        return np.asarray(return_matrix,dtype = np.float32)

    def save(self):
        pickle.dump(self.segment_trace, open(self.save_dir, 'wb'))
        # with open(self.save_dir, 'wb') as handle:
        #     pickle.dump(self.segment_trace, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __load(self):
        self.segment_trace = pickle.load(open(self.save_dir, 'rb'))
        # print('available bitrate level')
        # for k,v in self.segment_trace.items():
        #     print('{0}bps'.format(k))

if __name__ == '__main__':
    vlc = video_list_collector()
    vlc.seperate_trace()
    vlc.save()

