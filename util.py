
import numpy as np

def sample_Z(batch, z_dim , sampler = 'one_hot', num_class = 10, n_cat = 1, label_index = None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class*n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15*np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])


def sample_labelled_Z(batch, z_dim, sampler = 'one_hot', num_class = 10,  n_cat = 1, label_index = None):

    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((0.10 * np.random.randn(batch, z_dim - num_class*n_cat),
                                       np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])


def reshape_mnist(X):
    return X.reshape(X.shape[0], 28, 28, 1)


def clus_sample_Z(batch, dim_gen=20, dim_c=2, num_class = 10, label_index = None):

    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    batch_mat = np.zeros((batch, num_class* dim_c))
    for b in range(batch):
        batch_mat[b, label_index[b] * dim_c:(label_index[b] + 1) * dim_c] = np.random.normal(loc = 1.0, scale = 0.05, size = (1, dim_c))
    return np.hstack((0.10 * np.random.randn(batch, dim_gen), batch_mat))


def clus_sample_labelled_Z(batch, dim_gen=20, dim_c=2, num_class = 10, label_index = None):
    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    batch_mat = np.zeros((batch, num_class*dim_c))
    for b in range(batch):
        batch_mat[b, label_index[b] * dim_c:(label_index[b] + 1) * dim_c] = np.random.normal(loc=1.0, scale = 0.05, size = (1, dim_c))
    return label_index, np.hstack((0.10 * np.random.randn(batch, dim_gen), batch_mat))



def sample_info(batch, z_dim, sampler = 'one_hot', num_class = 10,  n_cat = 1, label_index = None):
    if sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack(
            (np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((np.random.randn(batch, z_dim - num_class*n_cat),
                                       np.tile(np.eye(num_class)[label_index], (1, n_cat))))


if __name__=='__main__':

    l = sample_Z(10, 22, 'mul_cat', 10, 2)
    print(l)

