import os
import Plotter_Helper as ph
import matplotlib.pyplot as plt
import torchvision.models
import torchsummary
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as Data
import adan
import relobralo
from pytorchtools import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dimension reduction net
class DRN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DRN, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        # x = self.Dense1(X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.features = input_dim
        # 每个工况的统计出现频率
        self.oce = torch.tensor([24.92, 14.95, 14.94, 15.16, 15, 15.02], dtype=torch.float32).to(device)
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
            nn.Linear(10, 6),
        )

    def forward(self, X):
        x = self.dnn(X)
        x = x * self.oce
        return x.sum(dim=1)


class PINN:
    def __init__(self, X, RUL, X_test, RUL_test, hidden_dim, derivatives_order, lr, batch_size, coef):
        self.X = torch.tensor(X[0:49072, :], dtype=torch.float32).to(device)
        self.RUL = torch.tensor(RUL[0:49072], dtype=torch.float32).to(device)
        # self.X_valid = X[49072:, :]
        # self.RUL_valid = RUL[49072:]
        # self.valid_process()
        self.X_valid = torch.tensor(X[49072:, :], dtype=torch.float32).to(device)
        self.RUL_valid = torch.tensor(RUL[49072:], dtype=torch.float32).to(device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        self.RUL_test = torch.tensor(RUL_test, dtype=torch.float32).to(device)

        self.hidden_dim = hidden_dim
        self.order = derivatives_order
        self.input_dim = 1 + self.hidden_dim * (self.order + 1)
        self.lr = lr
        self.batch_size = batch_size
        self.coef = coef

        self.drn = DRN(self.X.shape[1] - 1, self.hidden_dim).to(device)
        self.mlp = MLP(self.hidden_dim + 1).to(device)
        self.mlp.train()
        self.hnn = DRN(self.input_dim, 1).to(device)
        self.optim = adan.Adan(params=[{'params': self.drn.parameters()},
                                       {'params': self.mlp.parameters()},
                                       {'params': self.hnn.parameters()}
                                       ],
                               lr=self.lr)
        # 随机回顾的随机数，用于自适应损失函数可复现
        self.r = (np.random.uniform(size=int(1e8)) < 0.9999).astype(int).astype(np.float32)
        torchsummary.summary(self.drn, (1, self.X.shape[1] - 1))
        torchsummary.summary(self.mlp, (1, self.hidden_dim + 1))
        torchsummary.summary(self.hnn, (1, self.input_dim))

    def valid_process(self):
        xu = np.hstack((self.X_valid, self.RUL_valid.reshape(-1, 1)))
        ret = []
        for data in xu:
            if data[-1] < 125:
                ret.append(data)
        ret = np.array(ret)
        self.X_valid = torch.tensor(ret[:, 0:-1], dtype=torch.float32).to(device)
        self.RUL_valid = torch.tensor(ret[:, -1], dtype=torch.float32).to(device)

    # 评估分数函数
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

    # 用于预测RUL和输出隐藏状态
    def net_u(self, x, t):
        hidden = self.drn(x)
        hidden.requires_grad_(True)
        return self.mlp(torch.concat([hidden, t], dim=1)), hidden

    # 根据RUL和隐藏状态求各阶偏导数，完成PINN的残差项
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
        f = u_t - self.hnn(deri)
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
        # 早停止函数，20轮验证损失不下降就停止训练，同时保存验证损失最好的模型
        early_stopping = EarlyStopping(patience=20, path1='./Result/DRN.pth', path2='./Result/MLP.pth',
                                       path3='./Result/HNN.pth')
        for epoch in range(epochs):
            for step, (x, t, rul) in enumerate(loader):
                u, h = self.net_u(x, t)
                f = self.net_f(x, t)
                loss1 = torch.sqrt(MSE(u, rul))
                loss2 = torch.sqrt(MSE(f, torch.zeros(f.shape).to(device)))
                # 自适应损失函数返回损失项权重
                lambs = relobralo.relobralo(loss_u=loss1, loss_f=self.coef * loss2,
                                            epoch=epoch, step=step, T=0.1, rho=self.r[0])
                loss = lambs[0] * loss1 + lambs[1] * self.coef * loss2
                self.r = self.r[1:]
                # loss = loss1
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
            early_stopping(valid_loss, model1=self.drn, model2=self.mlp, model3=self.hnn)
            if early_stopping.early_stop:
                # print("Early stopping")
                break  # 跳出迭代，结束训练
        self.predict()

    def predict(self):
        MSE = nn.MSELoss()
        # 读取验证集性能最好的网络
        if not os.path.exists('./Result/DRN.pth'):
            torch.save(self.drn.state_dict(), './Result/DRN.pth')
        if not os.path.exists('./Result/MLP.pth'):
            torch.save(self.mlp.state_dict(), './Result/MLP.pth')
        self.drn.load_state_dict(torch.load('./Result/DRN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        pred, h = self.net_u(self.X_test[:, 0:-1], self.X_test[:, -1].reshape(-1, 1))
        rmse = torch.sqrt(MSE(pred, self.RUL_test))
        score = self.Score(pred, self.RUL_test)
        print('Test_RMSE: %.2f,   Score: %.1f' % (rmse, score))

    def plot_Train(self):
        self.drn.load_state_dict(torch.load('./Result/DRN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        x = self.X[0:20000, 0:-1]
        t = self.X[0:20000, -1].reshape(-1, 1)
        u, h = self.net_u(x, t)
        ph.Plot3D(hidden_state=h.cpu().detach().numpy(), RUL=self.RUL[0:20000].cpu().detach().numpy())

    def plot_Valid(self):
        self.drn.load_state_dict(torch.load('./Result/DRN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        u_valid, h_valid = self.net_u(self.X_valid[:, 0:-1], self.X_valid[:, -1].reshape(-1, 1))
        ph.Plot3D(hidden_state=h_valid.cpu().detach().numpy(), RUL=u_valid.cpu().detach().numpy())

    def plot_Test(self):
        self.drn.load_state_dict(torch.load('./Result/DRN.pth'))
        self.mlp.load_state_dict(torch.load('./Result/MLP.pth'))
        pred, h = self.net_u(self.X_test[:, 0:-1], self.X_test[:, -1].reshape(-1, 1))
        ph.Plot3D(hidden_state=h.cpu().detach().numpy(), RUL=pred.cpu().detach().numpy())
