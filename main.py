import os
import random
import PINN
import numpy as np
import torch
import data_preprocess as dp


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch()
    d = dp.CMAPSSDataset('C-MAPSS')
    u, x, t = d.get_train_data()
    fau = d.get_train_failure()
    x = np.hstack((x, t))
    u_test, x_test, t_test = d.get_test_data()
    X_test = np.hstack((x_test, t_test))
    u_i, x_i, t_i = d.get_test_unit_data(index=110)
    pinn = PINN.PINN(x, u, fau, X_test, u_test, hidden_dim=3, derivatives_order=2, lr=0.001, batch_size=128, coef=100)
    # pinn.train(1000)
    pinn.predict()
