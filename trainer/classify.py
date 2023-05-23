'''
Classify by ensembling transfered models
'''


import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable


PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/TransferClassifier/"
import sys
sys.path.append(PATH)
from trainer.transfer import transfer_from_all
from trainer.train_s import train_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class classify(train_all):
    def __init__(self):
        super(classify, self).__init__(path = PATH, all=False, batch_size=100)
        self.s_model_list = [self.load_model(i) for i in range(10)]

    def load_model(self, id):
        model_f = self.model_f.load_state_dict(torch.load(self.path+'model/f_task_transfer'+str(id)+'.pth', map_location=device))
        model_g = self.model_g.load_state_dict(torch.load(self.path+'model/g_task_transfer'+str(id)+'.pth', map_location=device))
        return [model_f, model_g]     
    
    def get_single_predict(self, model_f, model_g, image):
        # Test the model
        model_f.eval()
        model_g.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            
            fc = model_f(Variable(images).to(device)).data.cpu().numpy()
            f_mean = np.sum(fc,axis = 0) / fc.shape[0]
            fcp = fc - f_mean
            
            labellist = torch.eye(2)
            gc = model_g(Variable(labellist).to(device)).data.cpu().numpy()
            gce = np.sum(gc,axis = 0) / gc.shape[0]
            gcp = gc - gce
            fgp = np.dot(fcp, gcp.T)
            pred = np.argmax(fgp, axis = 1)
              
        return pred, fgp
    
    def get_pred(self, sample, label=None):

        image = sample.unsqueeze(0) # (3, 224, 224) -> (1, 3, 224, 224)

        with torch.no_grad():

            res_all = []
            for f, g in self.s_model_list:

                _, res = get_single_predict(f, g, image)
                res = res[0] / res[1]
                res_all.append(res)

            pred = np.argmax(np.array(res_all))

        
        istrue = pred == label if label != None else None
        
        return pred, istrue