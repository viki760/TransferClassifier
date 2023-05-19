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
    
    def get_ith_prediction(self, id):
        self.model_f.load_state_dict(torch.load(self.path+'model/f_task_transfer'+str(id)+'.pth', map_location=device))
        self.model_g.load_state_dict(torch.load(self.path+'model/g_task_transfer'+str(id)+'.pth', map_location=device))
        
        print('id = '+str(id))
        pred_i = self.test_model()
        return pred_i
    
    def test_model(self):
        # Test the model
        self.model_f.eval()
        self.model_g.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            acc = 0
            total = 0

            for images, labels in self.test_loader:

                labels = labels.numpy()
                fc = self.model_f(Variable(images).to(device)).data.cpu().numpy()
                f_mean = np.sum(fc,axis = 0) / fc.shape[0]
                fcp = fc - f_mean
                
                labellist = torch.eye(self.num_class)
                gc = self.model_g(Variable(labellist).to(device)).data.cpu().numpy()
                gce = np.sum(gc,axis = 0) / gc.shape[0]
                gcp = gc - gce
                fgp = np.dot(fcp, gcp.T)
                acc += (np.argmax(fgp, axis = 1) == labels).sum()
                total += len(images)

            acc = float(acc) / total
            print('Test Accuracy of the model on the 1000 test images: {} %'.format(100 * acc))
        
        return acc
    


