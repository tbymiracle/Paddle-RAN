from __future__ import print_function, division
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
# from torch.utils.data import Dataset, DataLoader
from paddle.io import DataLoader
from paddle.vision import datasets, transforms
import numpy as np
# from torchvision import transforms, datasets, models
# import os
# import cv2
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.ran_paddle import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

model_file = '/home/aistudio/work/RAN/RAN/model_92_sgd.pdparams'


# for test
def test(model, test_loader, btrain=False, model_file='model_92_sgd.pdparams'):
    # Test
    if not btrain:
        model.load_dict(paddle.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for j,(images, labels) in enumerate(test_loader):
        # images = Variable(images.cuda())
        # labels = Variable(labels.cuda())
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = paddle.topk(outputs, 1, 1)
        total += labels.shape[0]
        # print(predicted.astype('float32') == labels.astype('float32'))
        # correct += (predicted.astype('float32') == labels.astype('float32')).sum()
        correct += (predicted.squeeze() == labels).astype(int).sum()  
        c = (predicted.squeeze() == labels).astype(int).squeeze()
        for i in range(20):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    # print(total)
    # print(float(correct))
    # print(100 * class_correct[0] / class_total[0])
    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct)/total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total


# Image Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
    # transforms.Scale(224),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.ToTensor()
])
# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset
train_dataset = datasets.Cifar10(data_file='/home/aistudio/work/RAN/RAN/cifar-10-python.tar.gz',
                               mode = 'train',
                               transform=transform,
                               download=True)

test_dataset = datasets.Cifar10(data_file='/home/aistudio/work/RAN/RAN/cifar-10-python.tar.gz',
                              mode='test',
                              transform=test_transform)

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=64, # 64
                                           shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=20,
                                          shuffle=False, num_workers=8)
print(len(train_loader),len(test_loader))
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = ResidualAttentionModel()
#print(model)

lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()

#print(model.parameters())
optimizer = optim.Momentum(parameters=model.parameters(), momentum=0.9, use_nesterov=True, weight_decay=0.0001)
is_train = True
is_pretrain = False
acc_best = 0
total_epoch = 300
if is_train is True:
    if is_pretrain == True:
        model.load_dict((paddle.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            # images = Variable(images.cuda())
            # # print(images.data)
            # labels = Variable(labels.cuda())
            # print(images.shape)
            images = images.cuda()
            labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.clear_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("hello")
            if (i+1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss))

        print('the epoch takes time:',time.time()-tims)
        print('evaluate test set:')
        acc = test(model, test_loader, btrain=True)
        if acc > acc_best:
            acc_best = acc
            print('current best acc,', acc_best)
            paddle.save(model.state_dict(), 'best_model_92_sgd.pdparams')
        # Decaying Learning Rate
        if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
            lr /= 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    # Save the Model
    paddle.save(model.state_dict(), 'last_model_92_sgd.pdparams')

else:
    test(model, test_loader, btrain=False)

