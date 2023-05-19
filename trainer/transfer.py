'''
Transfer using finetuning based on source model f from train_s
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
from trainer.train_s import train_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


DATA_PATH = PATH + "data/"
MODEL_PATH = PATH + "model/"
SAVE_PATH = PATH + "result/"
N_TASK = 5


def load_data_id(id, batch_size=100):
    '''
    task_id    range(10)
    batch_size 100
    '''
    #! 没有 1 vs rest 数据需要添加处理

    x_train = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_train.npy").transpose((0,3,1,2))).to(torch.float32)
    y_train = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_train.npy"))
    trainloader = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    
    x_test = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_test.npy").transpose((0,3,1,2))).to(torch.float32)
    y_test = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_test.npy"))
    testloader = torch.utils.data.DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

class transfer_from_all(train_all):
    def __init__(self, id, batch_size=100):
        super(transfer_from_all, self).__init__(path = PATH, all=False, batch_size=batch_size)
        self.id = id
        self.model_f.load_state_dict(torch.load(MODEL_PATH+'f_task_all.pth', map_location=device))
    
    def load_data(self, batch_size):
        return load_data_id(batch_size = batch_size, id = self.id)
    
    def finetune(self, train_f=False, num_epochs = 20, lr = 0.0001, print_loss=True):
        
        self.model_g.train()

        if train_f:
            self.model_f.train()
            optimizer_fg = torch.optim.Adam(list(self.model_f.parameters())+list(self.model_g.parameters()), lr=lr)
        else:
            self.model_f.eval()
            optimizer_fg = torch.optim.Adam(list(self.model_g.parameters()), lr=lr)

        total_step = len(self.train_loader)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                
                labels_one_hot = torch.zeros(len(labels), self.num_class).scatter_(1, labels.view(-1,1), 1)

                # Forward pass
                optimizer_fg.zero_grad()
                f = self.model_f(Variable(images).to(device))
                g = self.model_g(Variable(labels_one_hot).to(device))

                loss = (-2)*self.corr(f,g) + 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum() + self.cov_trace(f,g)

                loss.backward()

                optimizer_fg.step()
                
                if print_loss and (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        print('Finished Training')
    
    def tuning(self):
        pass

    def save_model(self, train_f = False, dir = 'model/', test = True):

        print('-----------------------Save Model for id = '+str(self.id)+'--------------------')
        self.finetune(train_f = train_f)

        if test:
            self.test_model()

        if train_f:
            print('*finetuning both f and g*')
            save_path_f = self.path + dir + 'f_task_transfer'+str(self.id)+'.pth'
            torch.save(self.model_f.state_dict(), save_path_f)

        save_path_g = self.path + dir + 'g_task_transfer'+str(self.id)+'.pth'
        torch.save(self.model_g.state_dict(), save_path_g)




if __name__ == '__main__':
    batch_size = 10
    num_epochs = 20
    lr = 0.0001
    num_class = 10

    for id in range(num_class):
        train_id = transfer_from_all(id, batch_size=100)
        #! need to self define tuning parameters
        train_id.tuning()
        train_id.save_model()
        