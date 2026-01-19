import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "data/val/bees/44105569_16720a960c.jpg"
img_np = cv2.imread(img_path)

trans_img_to_tensor = transforms.ToTensor()
img_tensor = trans_img_to_tensor(img_np)

writer = SummaryWriter("logs")
writer.add_image("bees",img_tensor)
writer.close()