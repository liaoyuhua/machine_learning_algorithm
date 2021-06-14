from torch import nn
import torch

class ND(nn.Module):
    def __init__(self, n, units=10, noise=0.001):
        super(ND, self).__init__()
        self.wave = nn.Linear(1, n)
        self.unit_linear = nn.Linear(1, units)
        self.unit_softplus = nn.Linear(1, units)
        self.unit_sigmoid = nn.Linear(1, units)
        self.fc = nn.Linear(n + 3 * units, 1)
        params = dict(self.named_parameters())
        params['wave']

    def forward(self, X):
        sin = torch.sin(self.wave(X))
        linear = self.unit_linear(X)
        softplus = nn.Softplus()(self.unit_softplus(X))
        sigmoid = nn.Sigmoid()(self.unit_sigmoid(X))
        combinded = torch.cat([sin, linear, softplus, sigmoid], dim=1)
        out = self.fc(combinded)
        return out

# define our model
def model(X, Y, n, units, noise, loss, epochs, batch_size):
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
params = {'units': None
        , 'noise': None
        , 'epochs': None
        , 'batch_size': None
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
                 , params['batch_size'])
