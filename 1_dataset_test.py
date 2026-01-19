from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.image_path = os.listdir(self.path)
        
    def __getitem__(self, index):
        img_name = self.image_path[index]
        item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        image = Image.open(item_path)
        lable = self.label_dir
        return image,lable
    
root_dir = "dataset/train"
label_dir = "moon"
my_12_dataset = MyDataset(root_dir,label_dir)
img,lable = my_12_dataset[0]
img.show()