import matplotlib
from matplotlib import pyplot as plt


def Plot3D(hidden_state, RUL, style='Spectral'):
    x1 = hidden_state[:, 0]
    x2 = hidden_state[:, 1]
    x3 = hidden_state[:, 2]
    minr = min(RUL)
    maxr = max(RUL)
    cmap = [plt.get_cmap(style, 125)(int(float(i - minr) / (maxr - minr) * 125)) for i in RUL]
    fig = plt.figure("Hidden_state-RUL")
    ax3d = plt.gca(projection="3d")

    ax3d.set_xlabel('x1', fontsize=14)
    ax3d.set_ylabel('x2', fontsize=14)
    ax3d.set_zlabel('x3', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.set_cmap(plt.get_cmap(style, 125))
    im = ax3d.scatter(x1, x2, x3, c=cmap, marker='.')
    cb = fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x, pos: int(x * (maxr - minr) + minr)))
    cb.set_label(label='RUL Values', fontdict={"family": "Times New Roman", "size": 13})
    ax3d.view_init(15, 128)
    plt.show()


def PlotFailure(hidden_state, fau):
    x1 = hidden_state[:, 0]
    x2 = hidden_state[:, 1]
    x3 = hidden_state[:, 2]
    x0_1, x0_2, x0_3 = [], [], []
    x1_1, x1_2, x1_3 = [], [], []
    for i in range(fau.shape[0]):
        if fau[i] == 0:
            x0_1.append(x1[i])
            x0_2.append(x2[i])
            x0_3.append(x3[i])
        else:
            x1_1.append(x1[i])
            x1_2.append(x2[i])
            x1_3.append(x3[i])
    fig = plt.figure("Hidden_state-RUL")
    ax3d = plt.gca(projection="3d")

    ax3d.set_xlabel('x1', fontsize=14)
    ax3d.set_ylabel('x2', fontsize=14)
    ax3d.set_zlabel('x3', fontsize=14)
    plt.tick_params(labelsize=10)
    ax3d.scatter(x0_1, x0_2, x0_3, c='#ff9098', marker='.', label='failure mode 0')
    ax3d.scatter(x1_1, x1_2, x1_3, c='#6399AD', marker='.', label='failure mode 1')
    ax3d.view_init(15, 128)
    plt.legend()
    plt.show()
