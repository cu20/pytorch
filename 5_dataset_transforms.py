from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = datasets.CIFAR10("./data", train = True, transform=dataset_transform, download=True)
test_set = datasets.CIFAR10("./data", train = False, transform=dataset_transform, download=True)

# print(train_set[0])
# img , label = train_set[0]
# img.show()
# print(test_set.classes[label])
# print(train_set[0]) #转变为tensor的训练集
#试一试用tensorboard open it
writer = SummaryWriter("logs")
for i in range(0,10):
    img , label = train_set[i]
    print(label)
    writer.add_image("train_set",img, global_step = i)
writer.close