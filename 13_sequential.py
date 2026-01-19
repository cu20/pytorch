import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

class LiPanYue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Sequential(
            nn.Conv2d(3,32,5,1,2),  # stride=1 而不是 5
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),  # stride=1 而不是 5
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),  # stride=1 而不是 5
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,10)
        )
    
    def forward(self, x):
        output = self.model1(x)
        return output

lpy = LiPanYue()
print(lpy)
input = torch.ones((64,3,32,32))
output = lpy(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(lpy,input)
writer.close()