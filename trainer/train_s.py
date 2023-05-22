'''
training of source model
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/TransferClassifier/"


class Net_f(nn.Module):
    def __init__(self):
        super(Net_f, self).__init__()
        googlenet = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        self.feature=torch.nn.Sequential(*list(googlenet.children())[0:18])
        self.fc1 = nn.Linear(1024,32)
        self.fc2 = nn.Linear(32,10)
        self.BN = nn.BatchNorm1d(10)

    def forward(self,x):
        out=self.feature(x)
        out=out.view(-1,1024)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        out=self.BN(out)

        return out     

class Net_g(nn.Module):
    def __init__(self,num_class=10, dim=10):
        super(Net_g, self).__init__()

        self.fc=nn.Linear(num_class, dim)

    def forward(self,x):
        out=self.fc(x)

        return out


def load_data_all(batch_size=100):
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes



class train_all():

    def __init__(self, path = PATH, all=True, batch_size=100):

        self.num_class = 10 if all else 2
        self.train_loader, self.test_loader, self.classes = self.load_data(batch_size=batch_size)
        self.model_f = Net_f().to(device)
        self.model_g = Net_g(num_class = num_class).to(device)
        self.path = path

    def load_data(self, batch_size):
        return load_data_all(batch_size=batch_size)
    
    def corr(self, f, g):
        k = torch.mean(torch.sum(f*g,1))
        return k
        
    def cov_trace(self, f, g):
        cov_f = torch.mm(torch.t(f),f) / (f.size()[0]-1.)
        cov_g = torch.mm(torch.t(g),g) / (g.size()[0]-1.)
        return torch.trace(torch.mm(cov_f, cov_g))

    def train(self, num_epochs = 20, lr = 0.0001, print_loss=True) -> None:

        self.model_f.train()
        self.model_g.train()

        optimizer_fg = torch.optim.Adam(list(self.model_f.parameters())+list(self.model_g.parameters()), lr=lr)

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
    
    def tuning(self):
        pass
        
    def save_model(self, dir = 'model/', test = True):

        self.train()

        if test:
            self.test_model()

        save_path_f = self.path + dir + 'f_task_all.pth'
        torch.save(self.model_f.state_dict(), save_path_f)
        save_path_g = self.path + dir + 'g_task_all.pth'
        torch.save(self.model_g.state_dict(), save_path_g)
        


if __name__ == '__main__':
    # id = sys.argv[1]
    batch_size = 12
    num_epochs = 20
    lr = 0.0001
    num_class = 10

    train_s = train_all(path = PATH, batch_size=batch_size)
    #! need to self define tuning parameters
    train_s.tuning()
    train_s.save_model()