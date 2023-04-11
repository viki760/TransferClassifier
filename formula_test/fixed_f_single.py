

# calculate optimized g with fixed f
# calculate h score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
import loading
from fixed_f import fg
from OTCE import OTCE
import json


class transfer_fg(fg):

    '''
    calculation with fixed feature extractor with transfer
    '''

    def __init__(self, cfg, t_ids, s_ids, alpha):
        
        super(transfer_fg, self).__init__(cfg=cfg, t_id=t_ids)  

        self.s_ids = [s_ids] if isinstance(s_ids, int) else s_ids

        self.alpha = np.array(alpha) if isinstance(alpha, int) else alpha


    # def load_source(self, id):
    #     data = torch.load(f"{self.load_path}test{id}.pt")
    #     images, labels, f, g, n_label = data["x"], data["y"], data["f"], data["g"], data["n_label"]

    def load_for_id_with_source(self, id):
        
        # load task data and model
        self.read_from_load(id)
        self.model_f, self.model_g = loading.load_model(path = self.model_path, id = id)

        
        # load source data and model
        self.source_data = {}
        for s_id in self.s_ids:
            self.source_data[s_id] = torch.load(f"{self.load_path}test{s_id}.pt")

        self.s_f_list = np.cat([task["f"] for _, task in self.source_data.items()])
        self.s_g_list = torch.cat([task["g"] for _, task in self.source_data.items()])
        self.s_x_list = torch.cat([task["x"] for _, task in self.source_data.items()])
        self.s_y_list = torch.cat([task["y"] for _, task in self.source_data.items()])
        


    
    def empirical_transfer(self, t_id, s_id):
        pass 

    def get_g(self):

        # g = (1-alpha) * g + alpha * g_s

        # expectation and normalization of f and g
        n_f = self.normalize(self.f)
        # n_g = self.normalize(self.g)

        gamma_f = n_f.T.dot(n_f) / n_f.shape[0]

        ce_f = self. get_conditional_exp()
        
        ce_f_s = np.array([self.get_conditional_exp(self.s_x_list[s_id], self.s_y_list[s_id], self.s_f_list[s_id]) for s_id in self.s_ids])

        g_y_hat = np.linalg.inv(gamma_f).dot(((1-np.sum(self.alpha)) * ce_f + self.alpha.dot(ce_f_s)).T).T        
        
        g_rand = np.random.random(g_y_hat.shape)

        return g_rand, g_y_hat
    
    def get_accuracy_with_f(self):
        gc = self.s_tr_g_train()
        acc = 0
        total = 0

        for images, labels in self.test_data:

            labels= labels.numpy()
            fc=self.model_f_tr(Variable(images).to(self.device)).data.cpu().numpy()
            f_mean=np.sum(fc,axis=0)/fc.shape[0]
            fcp=fc-f_mean
            
            gce=np.sum(gc,axis=0)/self.n_label
            gcp=gc-gce
            fgp=np.dot(fcp,gcp.T)
            acc += (np.argmax(fgp, axis = 1) == labels).sum()
            # print(np.where(np.argmax(fgp, axis = 1) != labels))
            total += len(images)

        acc = float(acc) / total
        return acc
    
    def get_OTCE(self):
        # n_dim = self.data.shape
        return np.array([OTCE(self.s_x_list[s_id], self.s_y_list[s_id], self.images, self.labels) for s_id in self.s_ids]) 

    def acc(self):
        "output accuracy dict for all g and target tasks"
        acc_all = {}
        for id in self.t_id:
            self.load_for_id(id)
            acc = [self.get_accuracy(g) for g in self.get_g()]
            otce = self.get_OTCE()
            acc_list = {
                "g_rand": acc[0],
                "g_net": acc[1],
                "otce": otce,
            }
            acc_all[id] = acc_list     
        return acc_all



if __name__ == '__main__':
    import time
    import hydra
    from omegaconf import DictConfig

    # DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    # MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/weight/"
    # SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/results/"

    # mylog = open(SAVE_PATH+'otce.txt', mode = 'a',encoding='utf-8')

    N_TASK = 21
    TASK_LIST = range(N_TASK)

    alpha = 0.4

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def run(cfg : DictConfig)->None:  
        cal = transfer_fg(cfg, t_id=0, s_id=1, alpha=alpha)

        acc = cal.acc()
        json.dumps(acc, indent=4, sort_keys=True)
        cal.save(acc, "accuracy_dict")
        # print(W,' ',ce, file=mylog)

    
    # mylog.close()
    # np.savetxt(SAVE_PATH+'single_acc_table_'+time.strftime("%m%d", time.localtime())+'_alpha='+str(alpha)+'_t='+str(t_id)+'.npy', acc)
