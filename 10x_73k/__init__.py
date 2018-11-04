import scipy.sparse
import scipy.io
import numpy as np

# Original Labels are within 0 to 9. But proper label mapping is required as there are 8 classes.

class DataSampler(object):
    def __init__(self):
        self.total_size = 73233
        self.train_size = 60000
        self.test_size = 13233
        self.X_train, self.X_test = self._load_gene_mtx()
        self.y_train, self.y_test = self._load_labels()


    def _read_mtx(self, filename):
        buf = scipy.io.mmread(filename)
        return buf

    def _load_gene_mtx(self):
        data_path = './data/10x_73k/sub_set-720.mtx'
        data = self._read_mtx(data_path)
        data = data.toarray()
        data = np.log2(data + 1)
        scale = np.max(data)
        data = data / scale

        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        data_train = data[indx[0:self.train_size], :]
        data_test = data[indx[self.train_size:], :]

        return data_train, data_test


    def _load_labels(self):
        data_path = './data/10x_73k/labels.txt'
        labels = np.loadtxt(data_path).astype(int)

        np.random.seed(0)
        indx = np.random.permutation(np.arange(self.total_size))
        labels_train = labels[indx[0:self.train_size]]
        labels_test = labels[indx[self.train_size:]]
        return labels_train, labels_test

       
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.train_size, size = batch_size)

        if label:
            return self.X_train[indx, :], self.y_train[indx].flatten()
        else:
            return self.X_train[indx, :]

    def validation(self):
        return self.X_train[-5000:,:], self.y_train[-5000:].flatten()

    def test(self):
        return self.X_test, self.y_test

    def load_all(self):
         return np.concatenate((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test))

