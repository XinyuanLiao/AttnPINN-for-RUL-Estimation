import matplotlib
from matplotlib import pyplot as plt


def Plot3D(hidden_state, RUL):
    x1 = hidden_state[:, 0]
    x2 = hidden_state[:, 1]
    x3 = hidden_state[:, 2]
    cmap = RUL
    fig = plt.figure("Hidden_state-RUL")
    ax3d = plt.gca(projection="3d")  # 创建三维坐标

    plt.title('Hidden_state-RUL', fontsize=20)
    ax3d.set_xlabel('x1', fontsize=14)
    ax3d.set_ylabel('x2', fontsize=14)
    ax3d.set_zlabel('x3', fontsize=14)
    plt.tick_params(labelsize=10)

    im = ax3d.scatter(x1, x2, x3, c=cmap, marker='.')
    fig.colorbar(im, ax=ax3d, orientation='vertical')
    plt.show()
