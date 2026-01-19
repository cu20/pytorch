import torch
import torchvision

#方法一，已经被弃用
# model = torch.load("vgg16.pth")
# print(model)

#方法二
vgg16 = torchvision.models.vgg16(pretrained = False)
vgg16.load_state_dict(torch.load("vgg16_dict.pth"))
print(vgg16)