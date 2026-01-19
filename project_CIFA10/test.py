import torch 
import torchvision
from PIL import Image
import os
from model import *


# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "tudui_dict_10.pth")

model = Tudui()
model.load_state_dict(torch.load(model_path))

image_path = os.path.join(script_dir, "airplane.png")
image = Image.open(image_path)
composing = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])

image = composing(image)
image = torch.reshape(image,(1,3,32,32))
model.eval()

with torch.no_grad():
    output = model(image)

print(output)

print(output.argmax(1))