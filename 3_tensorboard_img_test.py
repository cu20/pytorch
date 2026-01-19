from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

img_path = "data/val/bees/26589803_5ba7000313.jpg"
img = Image.open(img_path)
img_np = np.array(img)

writer = SummaryWriter("tensor_logs")
writer.add_image("bees", img_np,1, dataformats="HWC")
writer.close()