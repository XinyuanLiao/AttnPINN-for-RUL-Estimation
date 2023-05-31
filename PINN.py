import os
import Plotter_Helper as ph
import matplotlib.pyplot as plt
import torchsummary
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as Data
import adan
import relobralo
from pytorchtools import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# x-NN for feature extraction
class xNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(xNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


# deep hidden physics network
class DeepHPM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepHPM, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


# multilayer perceptron for mapping hidden states to six RUL predictions
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.features = input_dim
        params = torch.ones(6)
        params = torch.full_like(params, 10, requires_grad=True)
        self.params = nn.Parameter(params)
        self.dnn = nn.Sequential(
            nn.Linear(self.features, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 6)
        )

    def forward(self, X):
        x = self.dnn(X)
        x = x * self.params
        return x.sum(dim=1)


class PINN:
    def __init__(self, X, RUL, fau, X_test, RUL_test, hidden_dim, derivatives_order, lr, batch_size, coef):
        # train set
        self.X = torch.tensor(X[0:49072, :], dtype=torch.float32).to(device)
        self.RUL = torch.tensor(RUL[0:49072], dtype=torch.float32).to(device)
        # valid set
        self.X_valid = torch.tensor(X[49072:, :], dtype=torch.float32).to(device)
        self.RUL_valid = torch.tensor(RUL[49072:], dtype=torch.float32).to(device)
        # test set
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        self.RUL_test = torch.tensor(RUL_test, dtype=torch.float32).to(device)
        self.fau = fau

        self.hidden_dim = hidden_dim
        self.order = derivatives_order
        self.input_dim = 1 + self.hidden_dim * (self.order + 1)
        self.lr = lr
        self.batch_size = batch_size
        self.coef = coef

        self.xnn = xNN(self.X.shape[1] - 1, self.hidden_dim).to(device)
        self.mlp = MLP(self.hidden_dim + 1).to(device)
        self.mlp.train()
        self.deepHPM = DeepHPM(self.input_dim, 1).to(device)
        self.optim = adan.Adan(params=[{'params': self.xnn.parameters()},
                                       {'params': self.mlp.parameters()},
                                       {'params': self.deepHPM.parameters()}
                                       ],
                               lr=self.lr)
        # random number for random lookback, reproducible for adaptive loss function
        self.r = (np.random.uniform(size=int(1e8)) < 0.9999).astype(int).astype(np.float32)
        self.a = [1, 0, 0.999]
        self.l0 = [1, 1]
        self.l1 = [1, 1]
        self.lamb = [1, 1]
        torchsummary.summary(self.xnn, (1, self.X.shape[1] - 1))
        torchsummary.summary(self.mlp, (1, self.hidden_dim + 1))
        torchsummary.summary(self.deepHPM, (1, self.input_dim))

    # scoring function
    def Score(self, pred, true):
        score = 0
        for i in range(pred.shape[0]):
            h = pred[i] - true[i]
            if h >= 0:
                s = torch.exp(h / 10) - 1
            else:
                s = torch.exp(-h / 13) - 1
            score += s
        return score

    # predict RUL and hidden state
    def net_u(self, x, t):
        hidden = self.xnn(x)
        hidden.requires_grad_(True)
        return self.mlp(torch.concat([hidden, t], dim=1)), hidden

    # Find the partial derivatives of each order according to RUL and hidden state,
    # and complete the residual term of PINN
    def net_f(self, x, t):
        t.requires_grad_(True)
        u, h = self.net_u(x, t)
        u = u.reshape(-1, 1)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_h = [u]
        for i in range(self.order):
            u_ = torch.autograd.grad(
                u_h[-1], h,
                grad_outputs=torch.ones_like(u_h[-1]),
                retain_graph=True,
                create_graph=True
            )[0]
            u_h.append(u_)
        deri = h
        for data in u_h:
            deri = torch.concat([deri, data], dim=1)
        f = u_t - self.deepHPM(deri)
        return f

    def train(self, epochs):
        x = self.X[:, 0:-1]
        t = self.X[:, -1].reshape(-1, 1)
        MSE = nn.MSELoss()
        dataset = Data.TensorDataset(x, t, self.RUL)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        # Early stop function, stop training if the verification loss does not decrease after 50 rounds,
        # and save the model with the best verification loss
        early_stopping = EarlyStopping(patience=50, path1='./Result/xNN.pth', path2='./Result/MLP.pth',
                                       path3='./Result/DeepHPM.pth')
        for epoch in range(epochs):
            for step, (x, t, rul) in enumerate(loader):
                u, h = self.net_u(x, t)
                f = self.net_f(x, t)
                loss1 = torch.sqrt(MSE(u, rul))
                loss2 = torch.sqrt(MSE(f, torch.zeros(f.shape).to(device)))
                # The adaptive loss function returns the loss term weights
                self.lamb = relobralo.relobralo(loss_u=loss1, loss_f=self.coef * loss2, alpha=self.a[0], l0=self.l0,
                                                l1=self.l1, lam=self.lamb, T=0.1, rho=self.r[0])
                loss = self.lamb[0] * loss1 + self.lamb[1] * self.coef * loss2
                if len(self.a) > 1:
                    self.a = self.a[1:]
                self.r = self.r[1:]
                losses = [loss1, self.coef * loss2]
                if epoch == 0 and step == 0:
                    self.l0 = losses
                self.l1 = losses
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            rul_valid, h_valid = self.net_u(self.X_valid[:, 0:-1], self.X_valid[:, -1].reshape(-1, 1))
            valid_loss = MSE(rul_valid, self.RUL_valid)
            if epoch % 1 == 0:
                print(
                    'It: %d,   Valid_RUL_RMSE: %.2f' %
                    (
                        epoch,
                        torch.sqrt(valid_loss)
                    )
                )
            early_stopping(valid_loss, model1=self.xnn, model2=self.mlp, model3=self.deepHPM)
            if early_stopping.early_stop:
                print("Early stopping")
                break  # Jump out of the iteration and end the training
        self.predict()

    # predict the test set and plot it
    def predict(self):
        MSE = nn.MSELoss()
        # The network with the best read validation set performance
        if not os.path.exists('./Result/xNN.pth'):
            torch.save(self.xnn.state_dict(), './Result/xNN.pth')
        if not os.path.exists('./Result/MLP.pth'):
            torch.save(self.mlp.state_dict(), './Result/MLP.pth')
        self.xnn.load_state_dict(torch.load('./Result/xNN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        pred, h = self.net_u(self.X_test[:, 0:-1], self.X_test[:, -1].reshape(-1, 1))
        rmse = torch.sqrt(MSE(pred, self.RUL_test))
        score = self.Score(pred, self.RUL_test)
        print('Test_RMSE: %.2f,   Score: %.1f' % (rmse, score))
        index = np.arange(248)
        rul_test = self.RUL_test.cpu().detach().numpy()
        pred_test = pred.cpu().detach().numpy()
        true_pred = np.hstack((rul_test.reshape(-1, 1), pred_test.reshape(-1, 1)))
        true_pred = sorted(true_pred, key=lambda x: x[0])
        true_pred = np.array(true_pred)
        pred = true_pred[:, 1]
        true = true_pred[:, 0]
        plt.plot(index, pred, color='#ff9098', marker='.', label='Pred')
        plt.plot(index, true, color='#6399AD', marker='.', label='True')
        plt.xlabel('Engine units after sorting', fontdict={"family": "Times New Roman", "size": 20})
        plt.ylabel('Remaining useful life', fontdict={"family": "Times New Roman", "size": 20})
        plt.legend()
        plt.show()

    # Print the predictions for an engine in the test set
    def plotUnit(self, u, x, t):
        u = torch.tensor(u, dtype=torch.float32).to(device)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        t = torch.tensor(t, dtype=torch.float32).to(device)
        self.xnn.load_state_dict(torch.load('./Result/xNN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        u_pred, h = self.net_u(x, t.reshape(-1, 1))
        plt.plot(t.cpu().detach().numpy(), u_pred.cpu().detach().numpy(), c='#ff9098', marker='.', label='Pred')
        plt.plot(t.cpu().detach().numpy(), u.cpu().detach().numpy(), c='#6399AD', marker='.', label='True')
        plt.xlabel('Cycles', fontdict={"family": "Times New Roman", "size": 13})
        plt.ylabel('Remaining useful life', fontdict={"family": "Times New Roman", "size": 13})
        plt.legend()
        plt.show()

    # plot the train set processed by x-NN in the hidden state space
    def plot_Train(self):
        self.xnn.load_state_dict(torch.load('./Result/xNN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        x = self.X[0:20000, 0:-1]
        t = self.X[0:20000, -1].reshape(-1, 1)
        u, h = self.net_u(x, t)
        ph.Plot3D(hidden_state=h.cpu().detach().numpy(), RUL=self.RUL[0:20000].cpu().detach().numpy())

    def plot_Valid(self):
        self.xnn.load_state_dict(torch.load('./Result/xNN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        u_valid, h_valid = self.net_u(self.X_valid[:, 0:-1], self.X_valid[:, -1].reshape(-1, 1))
        ph.Plot3D(hidden_state=h_valid.cpu().detach().numpy(), RUL=u_valid.cpu().detach().numpy())

    def plot_Test(self):
        self.xnn.load_state_dict(torch.load('./Result/xNN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        pred, h = self.net_u(self.X_test[:, 0:-1], self.X_test[:, -1].reshape(-1, 1))
        ph.Plot3D(hidden_state=h.cpu().detach().numpy(), RUL=pred.cpu().detach().numpy())

    # plot the failure mode in the hidden state space
    def plot_Fa(self):
        self.xnn.load_state_dict(torch.load('./Result/xNN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        x = self.X[0:20000, 0:-1]
        t = self.X[0:20000, -1].reshape(-1, 1)
        u, h = self.net_u(x, t)
        ph.PlotFailure(hidden_state=h.cpu().detach().numpy(), fau=self.fau[0:20000])
