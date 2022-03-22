# my own functions and libraries
import mobilenet
import read

# import standard library
import torch
import math
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# unused package
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resnet34():
    layers = [3, 4, 6, 3]
    model = mobilenet.ResNet(mobilenet.BasicBlock, layers, channels=19, num_classes=3)
    return model

class EEGDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        return X, y

class Combine_ResNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Combine_ResNet, self).__init__()
        self.output_dim = output_dim
        self.resnet = resnet34()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size()[0], x.size()[1], int(math.sqrt(self.output_dim)), int(math.sqrt(self.output_dim)))
        x = self.resnet(x)
        return x



if __name__ == '__main__':

    dataset_path = '/home/yuzhe/DATASET/EEG/'
    X_train, X_test, y_train, y_test = read.read_data_folder(dataset_path)
    train_set = EEGDataset(X_train, y_train)
    test_set = EEGDataset(X_test, y_test)

    batch_size = 20
    shuffle_state = True

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_state)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_state)



    def model_train_and_test(num_epochs, model, train_loader, test_loader):

        for epoch in range(num_epochs):

            # Train the model
            train_num_correct = 0
            for i, (data, labels) in enumerate(train_loader):
                # gives batch data, normalize x when iterate data_loader
                b_x = data.type(torch.float) # batch x
                b_y = labels # batch y
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                loss = loss_func(output, b_y)

                pred = output.argmax(dim=1)
                train_num_correct += torch.eq(pred, b_y).sum().float().item()
                accuracy = train_num_correct/len(train_loader.dataset)

                # clear gradients for this training step
                optimizer.zero_grad()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()

                # print('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, num_epochs, i + 1, train_step, loss.item(), accuracy))

            pass

            # Test the model
            model.eval()
            eval_acc = 0
            eval_loss = 0
            for i, (data, labels) in enumerate(test_loader):
                # gives batch data, normalize x when iterate data_loader
                b_x = data.type(torch.float) # batch x
                b_y = labels # batch y
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                eval_loss += loss_func(output, b_y)

                pred = output.argmax(dim=1)
                eval_acc += torch.eq(pred, b_y).sum().float().item()

            pass

            print('Test: Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, num_epochs, eval_loss / len(test_loader.dataset), eval_acc/len(test_loader.dataset)))

        pass


    my_model = Combine_ResNet(200, 32*32).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=0.001)

    num_epochs = 100
    model_train_and_test(num_epochs, my_model, train_loader, test_loader)

