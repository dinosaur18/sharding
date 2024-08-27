from accelerate import Accelerator
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DemoDS(Dataset):
    def __init__(self):
        super().__init__()
        self.x = torch.randn(160, 1)
        self.y = self.x.sin().square().neg()
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.x.size(0)


model = nn.Sequential(
    nn.Linear(1, 4),
    nn.GELU(),
    nn.Linear(4, 1)
)
opt = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ConstantLR(opt)
train_loader = DataLoader(DemoDS(), 16)

accelerator = Accelerator()

model, opt, train_loader, scheduler = accelerator.prepare(
    model, opt, train_loader, scheduler
)

for epoch in range(10000):
    for x, y in train_loader:
        opt.zero_grad()
        pred = model(x)

        loss = F.mse_loss(pred, y)
        accelerator.backward(loss)

        opt.step()
        scheduler.step()
        
        print(f'loss: {loss.detach().item():.4f}')
