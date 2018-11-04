import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
mnist = input_data.read_data_sets('./data/fashion')


class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def train(self, batch_size, label=False):
        if label:
           return mnist.train.next_batch(batch_size)
        else:
           return mnist.train.next_batch(batch_size)[0]

    def test(self):
        return mnist.test.images, mnist.test.labels

    def validation(self):
        return mnist.validation.images, mnist.validation.labels


    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):

        X_train = mnist.train.images
        X_val = mnist.validation.images
        X_test = mnist.test.images

        Y_train = mnist.train.labels
        Y_val = mnist.validation.labels
        Y_test = mnist.test.labels

        X = np.concatenate((X_train, X_val, X_test))
        Y = np.concatenate((Y_train, Y_val, Y_test))

        return X, Y.flatten()


class NoiseSampler(object):

    def __init__(self, z_dim = 100, mode='uniform'):
        self.mode = mode
        self.z_dim = z_dim
        self.K = 10

        if self.mode == 'mix_gauss':
            self.mu_mat = (1.0) * np.eye(self.K, self.z_dim)
            self.sig = 0.1

        elif self.mode == 'one_hot':
            self.mu_mat = (1.0) * np.eye(self.K)
            self.sig = 0.10


        elif self.mode == 'pca_kmeans':

            data_x = mnist.train.images
            feature_mean = np.mean(data_x, axis = 0)
            data_x -= feature_mean
            data_embed = PCA(n_components=self.z_dim, random_state=0).fit_transform(data_x)
            data_x += feature_mean
            kmeans = KMeans(n_clusters=self.K, random_state=0)
            kmeans.fit(data_embed)
            self.mu_mat = kmeans.cluster_centers_
            shift = np.min(self.mu_mat)
            scale = np.max(self.mu_mat - shift)
            self.mu_mat = (self.mu_mat - shift)/scale
            self.sig = 0.10


    def __call__(self, batch_size, z_dim):
        if self.mode == 'uniform':
            return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
        elif self.mode == 'normal':
            return 0.15*np.random.randn(batch_size, z_dim)
        elif self.mode == 'mix_gauss':
            k = np.random.randint(low = 0, high = self.K, size=batch_size)
            return self.sig*np.random.randn(batch_size, z_dim) + self.mu_mat[k]
        elif self.mode == 'pca_kmeans':
            k = np.random.randint(low=0, high=self.K, size=batch_size)
            return self.sig * np.random.randn(batch_size, z_dim) + self.mu_mat[k]
        elif self.mode == 'one_hot':
            k = np.random.randint(low=0, high=self.K, size=batch_size)
            return np.hstack((self.sig * np.random.randn(batch_size, z_dim-self.K), self.mu_mat[k]))



if __name__=='__main__':

     data_x = mnist.train.images
     from sklearn.decomposition import PCA
     from sklearn.cluster import KMeans
     import pdb
     mu = np.mean(data_x, axis = 0)
     data_x -= mu
     print('Computing PCA ...')
     data_embed = PCA(n_components=10, random_state=0).fit_transform(data_x)
     print('Done !')
     data_x += mu

     print('Computing kmeans ...')
     kmeans = KMeans(n_clusters=4, random_state=0)
     kmeans.fit(data_embed)
     print('Done !')
     mu_mat = kmeans.cluster_centers_
     shift = np.min(mu_mat)
     scale = np.max(mu_mat - shift)
     mu_mat = (mu_mat - shift)/scale
     print(mu_mat.shape)
     print("Done !")



