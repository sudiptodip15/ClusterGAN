import numpy as np
# Original Labels are from 0 to 9.

class DataSampler(object):
     def __init__(self):
         finp_tr = './data/pendigit/pendigits.tra.txt'
         finp_tes = './data/pendigit/pendigits.tes.txt'
         data_tr = np.loadtxt(finp_tr, delimiter=',')
         self.X_train = data_tr[:, 0:16]
         self.X_train /= 100.0
         self.Y_train = data_tr[:, -1].astype(int)

         data_tes = np.loadtxt(finp_tes, delimiter=',')
         self.X_test = data_tes[:, 0:16]
         self.X_test /= 100.0
         self.Y_test = data_tes[:, -1].astype(int)

         self.X = np.concatenate((self.X_train, self.X_test))
         self.Y = np.concatenate((self.Y_train, self.Y_test)).astype(int)

         self.train_size = self.X_train.shape[0]
         self.test_size = self.X_test.shape[0]
         self.data_size = self.X.shape[0]

     def train(self, batch_size, label=False):
         indx = np.random.randint(low=0, high=self.train_size, size=batch_size)

         if label:
             return self.X_train[indx, :], self.Y_train[indx].flatten()
         else:
             return self.X_train[indx, :]

     def validation(self):
         return self.X_train[-1000:, :], self.Y_train[-1000:].flatten()

     def test(self):
        return self.X_test, self.Y_test.flatten()


     def load_all(self):
         return self.X, self.Y.flatten()











