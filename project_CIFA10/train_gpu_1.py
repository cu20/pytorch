import torch
from torch import nn
import torchvision
from model import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs_train")

train_data = torchvision.datasets.CIFAR10("../data",train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=False)
test_data = torchvision.datasets.CIFAR10("../data",train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=False)

len_train = len(train_data)
len_test = len(test_data)
# print("训练集数据长度：{}".format(len_train))
# print("测试集数据长度：{}".format(len_test))

#使用dataloader加载数据
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=64)

#创建网络模型
tudui = Tudui()
tudui = tudui.cuda()

#创建损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr = learning_rate)

#设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10

#训练
for i in range(epoch):
    print("--------------第{}轮训练开始-----------".format(i+1))

    #tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 1:
            #print("第{}次训练的损失值：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    #模型测试
    tudui.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_acc += accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集的正确率:{}".format(total_acc / len_test*64))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_acc / len_test, total_test_step)
    total_test_step += 1

    #保存每一轮训练的模型结果
    torch.save(tudui.state_dict(),"tudui_dict_{}.pth".format(i+1))

writer.close()