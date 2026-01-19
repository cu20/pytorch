#非线性激活层
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

input = torch.tensor([[1, 2, -1, 3],
                      [-1, 3, -2, 5]])
input = torch.reshape(input, (-1, 1, 2, 4))

# 正确传入 transform，并使用 DataLoader 类
test_set = torchvision.datasets.CIFAR10(
    "data",
    train=False,
    transform=torchvision.transforms.ToTensor()
)

datas = DataLoader(test_set, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
    def forward(self,input):
        output = self.relu1(input)
        return output
#对输入进行非线性激活
tudui = Tudui()
output = tudui(input)
print(output)

step = 0
writer = SummaryWriter("logs")
#试一试对图像进行非线性激活吧
for data in datas:
    imgs, label = data
    grid_input = torchvision.utils.make_grid(imgs)
    writer.add_image("input",grid_input,step)
    imgs = tudui(imgs)
    # 将批次图像合并成网格，避免批量维度导致格式不匹配
    grid = torchvision.utils.make_grid(imgs)
    writer.add_image("relu", grid, step)
    step = step + 1

writer.close()