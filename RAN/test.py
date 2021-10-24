from __future__ import print_function, division
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import time
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from dataloader import preprocessing, load_data

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def test(model, test_loader, model_file, btrain=False):
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

if __name__ == "__main__":
    # Best trained model
    best_model_file = './model_92_sgd.pdparams'

    # Load data
    data_file = './data/cifar-10-python.tar.gz'
    train_transform, test_transform = preprocessing()
    train_loader, test_loader = load_data(data_file, train_transform, test_transform)
    
    # Load model and Test
    model = ResidualAttentionModel()
    test(model = model, 
         test_loader = test_loader, 
         model_file = best_model_file, 
         btrain=False)

