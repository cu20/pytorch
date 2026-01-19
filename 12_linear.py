import torch
import torchvision

test_set = torchvision.datasets.CIFAR10("data", False, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=test_set,batch_size=64)

class Tudui(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(196608,10)
    
    def forward(self, x):
        output = self.linear1(x)
        return output
tudui = Tudui()

# DataLoader 不支持索引，需要使用迭代器获取数据
imgs, target = next(iter(dataloader))
print(imgs.shape)
print("reshape后:")
imgs = torch.reshape(imgs,(1,1,1,-1))
print(imgs.shape)
print("经过网络后:")
imgs = tudui(imgs)
print(imgs.shape)
print("flatten后:")
imgs = torch.flatten(imgs)
print(imgs.shape)