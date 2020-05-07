import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from pyemd import emd
import math


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":

    trn_dataset = datasets.MNIST('./mnist_data/',
                                 download=True,
                                 train=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),  # image to Tensor
                                     transforms.Normalize((0.1307,), (0.3081,))  # image, label
                                 ]))

    val_dataset = datasets.MNIST("./mnist_data/",
                                 download=False,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    batch_size = 64
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    net = Net()
    # print(net)

    dist_arrs = []
    for i in range(28 * 28):
        i_row, i_col = i // 28, i % 28
        tmp_dist = []
        for j in range(28 * 28):
            j_row, j_col = j // 28, j % 28
            dist = math.sqrt((i_row - j_row) ** 2. + (i_col - j_col) ** 2.)
            tmp_dist.append(dist)
        dist_arrs.append(tmp_dist)
    dist_matrix = np.array(dist_arrs, dtype=np.float64)

    for b in val_loader:
        img_1 = b[0][0].reshape(-1).cpu().numpy().astype(np.float64)
        img_2 = b[0][1].reshape(-1).cpu().numpy().astype(np.float64)

        # print(img_1)
        # print(dist_matrix)
        d = emd(img_1, img_2, distance_matrix=dist_matrix)
        print(d)
        # label
        print(b[1][0])
        break

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    for _ in range(10):
        # optimizer.zero_grad()  # zero the gradient buffers
        # output = net(input)
        # criterion = nn.MSELoss()
        # loss = criterion(output, target)
        # loss.backward()
        # optimizer.step()
        pass
