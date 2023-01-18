import numpy as np
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')


# Process the C-MAPSS Dataset
class CMAPSSDataset:
    def __init__(self, path):
        super(CMAPSSDataset).__init__()
        self.stds = None
        self.means = None
        self.train_data = None
        self.test_data = None
        self.path = path
        self.start = 125
        # Useful Sensor Number
        self.valid_monitor = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
        # operation condition
        self.oc_index = [2, 3, 4]

        # Process the train data
        self.train_data = np.loadtxt(self.path + '/train.txt')
        self.train_t = np.array(self.train_data[:, 1].reshape(-1, 1))
        self.train_engine_size = int(self.train_data[self.train_data.shape[0] - 1][0])
        self.train_rul = np.zeros(self.train_engine_size)
        for i in range(self.train_data.shape[0] - 1):
            if not self.train_data[i][0] == self.train_data[i + 1][0]:
                index = int(self.train_data[i][0])
                self.train_rul[index - 1] = self.train_data[i][1]
        self.train_rul[self.train_engine_size - 1] = self.train_data[self.train_data.shape[0] - 1][1]
        self.train_array = [[] for i in range(self.train_engine_size)]
        for data in self.train_data:
            index = int(data[0])
            data[1] = self.train_rul[index - 1] - data[1]
            self.train_array[index - 1].append(data)
        self.train_array = np.array(self.train_array)

        # Process the test data
        self.test_data = np.loadtxt(self.path + '/test.txt')
        self.test_t = np.array(self.test_data[:, 1].reshape(-1, 1))
        self.RUL = np.loadtxt(self.path + '/RUL.txt')
        self.test_engine_size = self.RUL.shape[0]
        self.test_rul = np.zeros(self.test_engine_size)
        for i in range(self.test_data.shape[0] - 1):
            if not self.test_data[i][0] == self.test_data[i + 1][0]:
                index = int(self.test_data[i][0])
                self.test_rul[index - 1] = self.test_data[i][1]
        self.test_rul[self.test_engine_size - 1] = self.test_data[self.test_data.shape[0] - 1][1]
        self.test_array = [[] for i in range(self.test_engine_size)]
        for data in self.test_data:
            index = int(data[0])
            data[1] = self.test_rul[index - 1] + self.RUL[index - 1] - data[1]
            self.test_array[index - 1].append(data)
        self.test_array = np.array(self.test_array)

    def get_test_id(self):
        return self.test_data[:, 0]

    def get_train_id(self):
        return self.train_data[:, 0]

    def get_train_oc(self):
        return data_nomalization(self.train_data[:, self.oc_index])

    def get_test_oc(self):
        oc = self.test_data[:, self.oc_index]
        oc = data_nomalization(oc)
        return oc

    def get_train_data(self):
        oc = self.get_train_oc()
        u = self.train_data[:, 1]
        t = self.train_t
        for i in range(u.shape[0]):
            if u[i] >= self.start:
                u[i] = self.start
        train = self.train_data[:, self.valid_monitor]
        train, self.means, self.stds = normal_train(oc, train)
        # train = data_nomalization(train)
        id = self.get_train_id()
        t_array = [[] for i in range(self.train_engine_size)]
        u_array = [[] for i in range(self.train_engine_size)]
        time_array = [[] for i in range(self.train_engine_size)]
        for i in range(id.shape[0]):
            t_array[int(id[i]) - 1].append(train[i, :])
            u_array[int(id[i]) - 1].append(u[i])
            time_array[int(id[i]) - 1].append(t[i])
        ret1, ret2, ret3 = [], [], []
        for i in range(self.train_engine_size):
            t_array[i] = np.array(t_array[i])
            u_array[i] = np.array(u_array[i])
            for data in t_array[i]:
                ret1.append(data)
            for data in u_array[i]:
                ret2.append(data)
            for data in time_array[i]:
                ret3.append(data)
        train = np.array(ret1)
        u = np.array(ret2)
        t = np.array(ret3)
        return u, train, t

    def get_test_unit_data(self, index):
        data = self.test_array[index-1]
        data = np.array(data)
        rul = data[:, 1]
        sensor_data = data[:, self.valid_monitor]
        oc = data[:, self.oc_index]
        oc = data_nomalization(oc)
        t = np.arange(data.shape[0])
        sensor_data = normal_test(oc, sensor_data, self.means, self.stds)
        return rul, sensor_data, t

    def get_test_data(self):
        rul = self.test_data[:, 1]
        data = self.test_data[:, self.valid_monitor]
        oc = self.get_test_oc()
        t = self.test_t
        for i in range(rul.shape[0]):
            if rul[i] >= self.start:
                rul[i] = self.start
        data = normal_test(oc, data, self.means, self.stds)
        # data = data_nomalization(data)
        id = self.get_test_id()
        t_array = [[] for i in range(self.test_engine_size)]
        rul_array = [[] for i in range(self.test_engine_size)]
        for i in range(id.shape[0]):
            t_array[int(id[i]) - 1].append(data[i, :])
            rul_array[int(id[i]) - 1].append(rul[i])
        ret = []
        ret_rul = []
        for i in range(self.test_engine_size):
            for j in range(len(self.valid_monitor)):
                t_array[i] = np.array(t_array[i])
        for data in t_array:
            ret.append(data[-1, :])
        for r in rul_array:
            ret_rul.append(r[-1])
        data = np.array(ret)
        rul = np.array(ret_rul)
        ret_t = []
        for i in range(t.shape[0] - 1):
            if t[i] > t[i + 1]:
                ret_t.append(t[i])
        ret_t.append(t[-1])
        t = np.array(ret_t)
        return rul, data, t


# Min-max nomalization featrue_range=(-1,1)
def data_nomalization(data, feature_range=(-1, 1)):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range)
    return min_max_scaler.fit_transform(data)


def normal_train(oc, x):
    label = np.zeros(x.shape[0])
    for i in range(label.shape[0]):
        label[i] = 10
    data = [[], [], [], [], [], []]
    m = [[], [], [], [], [], []]
    s = [[], [], [], [], [], []]
    for i in range(x.shape[0]):
        if np.abs(oc[i][0] - 1) <= 0.01 and np.abs(oc[i][1] - 1) <= 0.01 and np.abs(oc[i][2] - 1) <= 0.01:
            data[0].append(x[i, :])
            label[i] = 0
        elif np.abs(oc[i, 0] + 0.047) < 0.01 and np.abs(oc[i, 1] - 0.66) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[1].append(x[i, :])
            label[i] = 1
        elif np.abs(oc[i, 0] - 0.19) < 0.01 and np.abs(oc[i, 1] - 0.47) < 0.01 and np.abs(oc[i, 2] + 1) < 0.01:
            data[2].append(x[i, :])
            label[i] = 2
        elif np.abs(oc[i, 0] + 1) < 0.01 and np.abs(oc[i, 1] + 1) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[3].append(x[i, :])
            label[i] = 3
        elif np.abs(oc[i, 0] - 0.66) < 0.01 and np.abs(oc[i, 1] - 1) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[4].append(x[i, :])
            label[i] = 4
        elif np.abs(oc[i, 0] + 0.52) < 0.01 and np.abs(oc[i, 1] + 0.4) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[5].append(x[i, :])
            label[i] = 5
    for i in range(6):
        data[i] = np.array(data[i])
        for j in range(x.shape[1]):
            m[i].append(np.mean(data[i][:, j]))
            s[i].append(np.std(data[i][:, j]))
    m = np.array(m).reshape(6, 14)
    s = np.array(s).reshape(6, 14)
    label = np.array(label)
    for i in range(x.shape[0]):
        index = int(label[i])
        x[i] = (x[i] - m[index]) / s[index]
    return x, m, s


def normal_test(oc, x, means, stds):
    label = np.zeros(x.shape[0])
    for i in range(label.shape[0]):
        label[i] = 10
    data = [[], [], [], [], [], []]
    m = means
    s = stds
    for i in range(x.shape[0]):
        if np.abs(oc[i][0] - 1) <= 0.01 and np.abs(oc[i][1] - 1) <= 0.01 and np.abs(oc[i][2] - 1) <= 0.01:
            data[0].append(x[i, :])
            label[i] = 0
        elif np.abs(oc[i, 0] + 0.047) < 0.01 and np.abs(oc[i, 1] - 0.66) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[1].append(x[i, :])
            label[i] = 1
        elif np.abs(oc[i, 0] - 0.19) < 0.01 and np.abs(oc[i, 1] - 0.47) < 0.01 and np.abs(oc[i, 2] + 1) < 0.01:
            data[2].append(x[i, :])
            label[i] = 2
        elif np.abs(oc[i, 0] + 1) < 0.01 and np.abs(oc[i, 1] + 1) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[3].append(x[i, :])
            label[i] = 3
        elif np.abs(oc[i, 0] - 0.66) < 0.01 and np.abs(oc[i, 1] - 1) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[4].append(x[i, :])
            label[i] = 4
        elif np.abs(oc[i, 0] + 0.52) < 0.01 and np.abs(oc[i, 1] + 0.4) < 0.01 and np.abs(oc[i, 2] - 1) < 0.01:
            data[5].append(x[i, :])
            label[i] = 5
    for i in range(x.shape[0]):
        index = int(label[i])
        x[i] = (x[i] - m[index]) / s[index]
    return x
