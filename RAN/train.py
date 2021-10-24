from __future__ import print_function, division
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import time
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from dataloader import preprocessing, load_data

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# for test
def test(model, test_loader, model_file='./best_model_92_sgd.pdparams', btrain=False):
    if not btrain:
        model.load_dict(paddle.load(model_file))
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for j,(images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        # Top1 acc
        _, predicted = paddle.topk(outputs, 1, 1)
        total += labels.shape[0]
        correct += (predicted.squeeze() == labels).astype(int).sum()  
        c = (predicted.squeeze() == labels).astype(int).squeeze()
        for i in range(20):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct)/total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total


def train(model, model_file, total_epoch, train_loader, test_loader, is_pretrain=False):
    lr = 0.1  # 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Momentum(learning_rate=lr, parameters=model.parameters(), 
                        momentum=0.9, use_nesterov=True, weight_decay=0.0001)
    is_pretrain = False
    acc_best = 0
    if is_pretrain == True:
        model.load_dict((paddle.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.clear_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
            optimizer.set_lr(lr)
    # Save the Model
    paddle.save(model.state_dict(), 'last_model_92_sgd.pdparams')

if __name__ == "__main__":
    # Trained model
    model_file = '/home/aistudio/work/RAN/RAN/best_model_92_sgd.pdparams'
    
    # Load data
    data_file = '/home/aistudio/work/RAN/RAN/data/cifar-10-python.tar.gz'
    train_transform, test_transform = preprocessing()
    train_loader, test_loader = load_data(data_file, train_transform, test_transform)
    
    # Train
    model = ResidualAttentionModel()
    train(model = model, 
          model_file = model_file,
          total_epoch=300,
          train_loader = train_loader,
          test_loader = test_loader,
          is_pretrain=False)
    

