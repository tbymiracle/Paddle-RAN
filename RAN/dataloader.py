from paddle.io import DataLoader
from paddle.vision import datasets, transforms

def preprocessing():
# Image Preprocessing
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
        # transforms.Scale(224),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return train_transform, test_transform
# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset

def load_data(data_file, train_transform, test_transform):
    train_dataset = datasets.Cifar10(data_file,
                                mode = 'train',
                                transform=train_transform,
                                download=True)

    test_dataset = datasets.Cifar10(data_file,
                                mode='test',
                                transform=test_transform)

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=64, # 64
                                            shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset,
                                            batch_size=20,
                                            shuffle=False, num_workers=8)
    return train_loader, test_loader


