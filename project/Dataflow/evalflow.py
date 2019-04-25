
import numpy as np

class EvalFlow:
    
    def __init__(self, df):
        self.df = df
        self.piece_num = len(df.features)
        self.cur_pid = -1


    def init_index(self):
        self.idxs = [i for i in range(0, self.feature_len, self.df.timesteps)]
        
        common_shape = (self.df.b_sz, self.df.timesteps, self.df.feature_num)
        self.d_buffer = np.zeros(common_shape + (len(self.df.channels),))
        self.l_buffer = np.zeros(common_shape + (2 if self.df.mpe_only else 12, ))
        
        self.cur_iid = 0
        

    def __iter__(self):
        self.stop_next = False
        self.cur_pid += 1
        self.feature_len = len(self.df.features[self.cur_pid])
        self.init_index()
        #print("Current piece id: ", self.cur_pid)
        return self

    def __next__(self):
        if self.stop_next:
            #print("Next piece")
            raise StopIteration

        upper = min(self.cur_iid+self.df.b_sz, len(self.idxs))
        frms = range(self.cur_iid, upper)
        for ii, i in enumerate(frms):
            tid = self.idxs[i]
            x, y = self.df.get_feature(self.cur_pid, tid)

            self.d_buffer[ii] = x
            self.l_buffer[ii] = y[:,:,:2]
        self.cur_iid += len(frms)

        if self.cur_iid >= len(self.idxs):
            self.d_buffer[len(frms):] = 0
            self.l_buffer[len(frms):] = 0
            self.stop_next = True
        
        return self.d_buffer, self.l_buffer

    def __len__(self):
        return self.piece_num

#import sys
#sys.path.append("../..")
#from DataFlows import Maestro
if __name__ == "__main__":
    ds_path = "/media/whitebreeze/本機磁碟/maestro-v1.0.0"

    ds = Maestro(ds_path, "val")
    eval_flow = EvalFlow(ds)

    print("Length: ", len(eval_flow), len(ds.features))
    for _ in range(len(eval_flow)):
        for x, y in eval_flow:
            print(x.shape)
