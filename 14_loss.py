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

test_set = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(test_set,64)

loss = torch.nn.CrossEntropyLoss()
for data in dataloader:
    imgs,target = data
    output = lpy(imgs)
    result_loss = loss(output, target)
    print(result_loss)
    result_loss.backward()