import torch


def relobralo(loss_u, loss_f, alpha, l0, l1, lam, T=0.1, rho=0):
    alpha = alpha
    losses = [loss_u, loss_f]
    length = len(losses)
    length = torch.tensor(length, dtype=torch.float32)
    temp1 = torch.softmax(torch.tensor([losses[i] / (l1[i] * T + 1e-12) for i in range(len(losses))]), dim=-1)
    temp2 = torch.softmax(torch.tensor([losses[i] / (l0[i] * T + 1e-12) for i in range(len(losses))]), dim=-1)
    lambs_hat = torch.mul(temp1, length)
    lambs0_hat = torch.mul(temp2, length)
    lambs = [rho * alpha * lam[i] + (1 - rho) * alpha * lambs0_hat[i] + (1 - alpha) * lambs_hat[i] for i in
             range(len(losses))]
    return lambs
