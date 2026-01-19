from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("tensor_logs")

img_path = "data/val/bees/26589803_5ba7000313.jpg"
img = Image.open(img_path)
img_np = np.array(img)

for i in range(1,101):
    writer.add_scalar("y=2x",2*i,i) #前者为y，后者是x
    writer.add_scalar("y=x",i,i)
writer.add_image("bees", img_np,1, dataformats="HWC")
writer.close()

# 打开tensor_logs的命令行: tensorboard --logdir=tensor_logs --port=...