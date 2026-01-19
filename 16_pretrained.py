import torchvision

from torch import nn

#学会修改预训练模型

vgg16_notrained = torchvision.models.vgg16(pretrained = False)
vgg16_trained = torchvision.models.vgg16(pretrained = True)

print(vgg16_trained)

vgg16_trained.classifier.add_module("add_linear",nn.Linear(1000,10))

print(vgg16_trained)

print(vgg16_notrained)
vgg16_notrained.classifier[6] = nn.Linear(4096,10)
print(vgg16_notrained)