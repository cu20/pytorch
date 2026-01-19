import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

test_data = torchvision.datasets.CIFAR10("data",train = False,transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(test_data,batch_size=64)

class Tudui(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x
    
tudui = Tudui()
print(tudui)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs , targets = data
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)
    output = torch.reshape(output,(-1,3,30,30))
    img_grid_input = make_grid(imgs)
    img_grid_output = make_grid(output)
    writer.add_image("input",img_grid_input,step)
    writer.add_image("output",img_grid_output,step)
    step = step + 1

writer.close()