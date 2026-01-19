import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

test_data = torchvision.datasets.CIFAR10("data",train = False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset = test_data, batch_size = 64, num_workers = 0, drop_last = True)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,label = data
        img_grid = make_grid(imgs)
        writer.add_image("Epoch:{}".format(epoch),img_grid,step)
        step = step + 1
writer.close()