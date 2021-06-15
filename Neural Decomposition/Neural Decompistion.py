from torch import nn
import torch
import numpy as np

class ND(nn.Module):
    def __init__(self, n, units=10, noise=0.001):
        super(ND, self).__init__()
        # define layers of neural network
        self.wave = nn.Linear(1, n)
        self.unit_linear = nn.Linear(1, units)
        self.unit_softplus = nn.Linear(1, units)
        self.unit_sigmoid = nn.Linear(1, units)
        self.fc = nn.Linear(n + 3 * units, 1)
        # initialize weights
        params['wave.weight'].data = torch.from_numpy((2 * np.pi * np.floor(np.arange(n) / 2))[:, np.newaxis]).float()
        params['wave.bias'].data = torch.from_numpy(np.pi / 2 + np.arange(n) % 2 * np.pi / 2).float()
        params['unit_linear.weight'].data = torch.from_numpy(np.ones(shape=(units, 1)) + np.random.normal(size=(units, 1)) * noise).float()
        params['unit_linear.bias'].data = torch.from_numpy(np.random.normal(size=(units)) * noise).float()
        params['unit_softplus.weight'].data = torch.from_numpy(np.random.normal(size=(units, 1)) * noise).float()
        params['unit_softplus.bias'].data = torch.from_numpy(np.random.normal(size=(units)) * noise).float()
        params['unit_sigmoid.weight'].data = torch.from_numpy(np.random.normal(size=(units, 1)) * noise).float()
        params['unit_sigmoid.bias'].data = torch.from_numpy(np.random.normal(size=(units)) * noise).float()
        params['fc.weight'].data = torch.from_numpy(np.random.normal(size=(1, n + 3 * units)) * noise).float()
        params['fc.bias'].data = torch.from_numpy(np.random.normal(size=(1)) * noise).float()

    def forward(self, X):
        sin = torch.sin(self.wave(X))
        linear = self.unit_linear(X)
        softplus = nn.Softplus()(self.unit_softplus(X))
        sigmoid = nn.Sigmoid()(self.unit_sigmoid(X))
        combinded = torch.cat([sin, linear, softplus, sigmoid], dim=1)
        out = self.fc(combinded)
        return out

# define our model
def model(X, Y, n, units, noise, loss, epochs, batch_size, l1_ratio=0.01):
    """
    :param X: 训练features
    :param Y: 训练labels
    :param n:
    :param units:
    :param noise:
    :param loss:
    :param epochs:
    :param batch_size:
    :return:
    """
    nd = ND(n)
    optimizer = torch.optim.Adam(nd.parameters())
    for epoch in range(epochs):
        # clear gradient of last round
        optimizer.zero_grad()
        # forward
        outputs = nd.forward(X)
        loss = loss_fn(outputs, Y)
        # add L1 regularization
        loss += l1_ratio * torch.sum(torch.abs(dict(nd.named_parameters())['fc.weight']))
        # backward
        loss.backward()
        # update parameters
        optimizer.step()
        # display process of training
        if epoch % 100 == 0:
            print("Epoch {}, loss: {}".format(epoch, round(loss.data[0], 4)))
    return nd


# loss function
loss_fn = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam()

# hyperparameters
params = {'units': None,
          'noise': None,
          'epochs': None,
          'batch_size': None,
          'l1_ratio': None
}

# train model
n = None
X = None
Y = None

nn_model = model(X
                 , Y
                 , params['units']
                 , loss_fn
                 , params['epochs']
                 , params['batch_size']
                 , params['l1_ratio'])
