from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch


def create_dataloader(x,y,batch_size=64):
    x = torch.tensor(x, dtype=torch.float).cuda()
    y = torch.tensor(y, dtype=torch.long).cuda()
    data = TensorDataset(x, y)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader
