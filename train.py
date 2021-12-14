import time
import numpy as np
import torch
from torch import nn
from model import Classifier
from process import train_loader, val_loader, train_set, val_set, pre_loader
from torch.utils.tensorboard import SummaryWriter

model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 使用交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 选择优化器
num_epoch = 30

model_para = r"D:\personal-file\Work\CV-HW\data\model"
model_save = r"../model_save"
writer = SummaryWriter(model_para)

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    loss_compare = [0,]

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度归零
        train_pre = model(data[0].cuda())  #
        # print(type(train_pre))
        # print(type(data[1]))
        batch_loss = loss(train_pre, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pre.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()


        writer.add_scalars('acc', { 'train':train_acc / train_set.__len__(), 'val':val_acc / val_set.__len__()}, epoch)
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))
    loss_compare.append(val_acc / val_set.__len__())
    if loss_compare[-1] > loss_compare[-2]:
        torch.save(model, model_save)

pre_acc = 0.0
pre_loss = 0.0
model_use = torch.load(model_save)
for i, data in enumerate(pre_loader):
    pre_pred = model(data[0].cuda())
    batch_loss = loss(pre_pred, data[1].cuda())
    print("pre{}and{}True".format(np.argmax(pre_pred.cpu().data.numpy(), axis=1),data[1].numpy()))
    pre_acc += np.sum(np.argmax(pre_pred.cpu().data.numpy(), axis=1) == data[1].numpy())


print(pre_acc)