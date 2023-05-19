'''
generate 1 vs rest tasks

TODO 按照如下读取格式生成 task data
def load_data_id(id, batch_size=100):

    x_train = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_train.npy").transpose((0,3,1,2))).to(torch.float32)
    y_train = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_train.npy"))
    trainloader = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    
    x_test = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_test.npy").transpose((0,3,1,2))).to(torch.float32)
    y_test = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_test.npy"))
    testloader = torch.utils.data.DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

'''