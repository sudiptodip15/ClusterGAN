import os
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import metric
import util

tf.set_random_seed(0)

class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model, sampler, num_classes, dim_gen, n_cat,
                 batch_size, beta_reg):
        self.model = model
        self.data = data
        self.sampler = sampler
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.num_classes = num_classes
        self.dim_gen = dim_gen
        self.n_cat = n_cat
        self.batch_size = batch_size
        scale = 10.0
        self.beta_reg = beta_reg

        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim

        if sampler == 'mul_cat':
            self.clip_lim = [-0.6, 0.6]
        elif sampler == 'one_hot':
            self.clip_lim = [-0.6, 0.6]
        elif sampler == 'clus':
            self.clip_lim = [-1.0, 1.0]
        elif sampler == 'uniform':
            self.clip_lim = [-1.0, 1.0]
        elif sampler == 'normal':
            self.clip_lim = [-1.0, 1.0]
        elif sampler == 'mix_gauss':
            self.clip_lim = [-1.0, 2.0]
        elif sampler == 'pca_kmeans':
            self.clip_lim = [-2.0, 2.0]

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_net.vars)
        self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.g_loss, var_list=self.g_net.vars)

        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, num_batches=50000):

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        batch_size = self.batch_size
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        print('Training {} on {}, sampler = {}, z = {} dimension'.format(self.model, self.data, self.sampler, self.z_dim))


        for t in range(0, num_batches):
            d_iters = 5

            for _ in range(0, d_iters):
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            self.sess.run(self.g_adam, feed_dict={self.z: bz})

            if (t+1) % 100 == 0:
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                      (t+1, time.time() - start_time, d_loss, g_loss))

        self.recon_enc(timestamp, val=True)
        self.save(timestamp)

    def save(self, timestamp):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}'.format(self.data, timestamp, self.model, self.sampler,
                                                                 self.z_dim)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))

    def load(self, pre_trained=False, timestamp=''):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_z{}'.format(self.data, self.model, self.sampler, self.z_dim)
        else:
            if timestamp == '':
                print('Best Timestamp not provided !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}'.format(self.data, timestamp, self.model, self.sampler,
                                                                         self.z_dim)

        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        print('Restored model weights.')

    def _gen_samples(self, num_samples):

        batch_size = self.batch_size
        bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)  
        fake_samples = self.sess.run(self.x_, feed_dict = {self.z : bz})
        for t in range(num_samples // batch_size):
            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            samp = self.sess.run(self.x_, feed_dict = {self.z : bz})
            fake_samples = np.vstack((fake_samples, samp))

        print(' Generated {} samples .'.format(fake_samples.shape[0]))
        np.save('./Image_samples/{}/{}_{}_K_{}_gen_images.npy'.format(self.data, self.model, self.sampler, self.num_classes), fake_samples)


    def recon_enc(self, timestamp, val = True):

        if val:
            data_recon, label_recon = self.x_sampler.validation()
        else:
            data_recon, label_recon = self.x_sampler.test()
            #data_recon, label_recon = self.x_sampler.load_all()

        num_pts_to_plot = data_recon.shape[0]
        recon_batch_size = 1000
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))
        clip_lim = self.clip_lim

        print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))

        # Regularized Reconstruction objective

        self.recon_reg_loss = tf.reduce_mean(tf.abs(self.x - self.x_), 1) + \
                              self.beta_reg * tf.reduce_mean(tf.square(self.z[:, 0:self.dim_gen]), 1)
        self.compute_reg_grad = tf.gradients(self.recon_reg_loss, self.z)

        for b in range(int(np.ceil(num_pts_to_plot*1.0 / recon_batch_size))):

            if (b+1)*recon_batch_size > num_pts_to_plot:
               pt_indx = np.arange(b*recon_batch_size, num_pts_to_plot)
            else:
               pt_indx = np.arange(b*recon_batch_size, (b+1)*recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            num_backprop_iter = 5000
            num_restarts = self.num_classes
            seed_labels = np.tile(np.arange(self.num_classes), int(np.ceil(len(pt_indx) * 1.0 / self.num_classes)))
            seed_labels = seed_labels[0:len(pt_indx)]
            best_zhats = np.zeros(shape=(len(pt_indx), self.z_dim))
            best_loss = np.inf * np.ones(len(pt_indx))
            mu_mat = 1.0 * np.eye(self.num_classes)
            alg = 'adam'
            for t in range(num_restarts):
                print('Backprop Decoding [{} / {} ] ...'.format(t + 1, num_restarts))

                if self.sampler == 'one_hot':
                    label_index = (seed_labels + t) % self.num_classes
                    zhats = util.sample_Z(len(pt_indx), self.z_dim, self.sampler, num_class=self.num_classes,
                                          n_cat=1, label_index=label_index)
                elif self.sampler == 'mul_cat':
                    label_index = (seed_labels + t) % self.num_classes
                    zhats = util.sample_Z(len(pt_indx), self.z_dim, self.sampler, num_class=self.num_classes,
                                          n_cat=self.n_cat,
                                          label_index=label_index)
                elif self.sampler == 'mix_gauss':
                    label_index = (seed_labels + t) % self.num_classes
                    zhats = util.sample_Z(len(pt_indx), self.z_dim, self.sampler, num_class=self.num_classes,
                                          n_cat=0, label_index=label_index)
                else:
                    zhats = util.sample_Z(len(pt_indx), self.z_dim, self.sampler)
                if alg == 'adam':
                    beta1 = 0.9
                    beta2 = 0.999
                    lr = 0.01
                    eps = 1e-8
                    m = 0
                    v = 0
                elif alg == 'grad_descent':
                    lr = 1.00

                for i in range(num_backprop_iter):

                    L, g = self.sess.run([self.recon_reg_loss, self.compute_reg_grad],
                                         feed_dict={self.z: zhats, self.x: xtrue})

                    if alg == 'adam':
                        m_prev = np.copy(m)
                        v_prev = np.copy(v)
                        m = beta1 * m_prev + (1 - beta1) * g[0]
                        v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                        m_hat = m / (1 - beta1 ** (i + 1))
                        v_hat = v / (1 - beta2 ** (i + 1))
                        zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))

                    elif alg == 'grad_descent':
                        zhats += - lr * g[0]

                    zhats = np.clip(zhats, a_min=clip_lim[0], a_max=clip_lim[1])

                    if self.sampler == 'one_hot':
                        zhats[:, -self.num_classes:] = mu_mat[label_index, :]
                    elif self.sampler == 'mul_hot':
                        zhats[:, self.dim_gen:] = np.tile(mu_mat[label_index, :], (1, self.n_cat))

                change_index = best_loss > L
                best_zhats[change_index, :] = zhats[change_index, :]
                best_loss[change_index] = L[change_index]

            latent[pt_indx, :] = best_zhats
            print(' [{} / {} ] ...'.format(pt_indx[-1]+1, num_pts_to_plot))


        self._eval_cluster(latent, label_recon, timestamp, val)

    def _eval_cluster(self, latent_rep, labels_true, timestamp, val):

        km = KMeans(n_clusters=max(self.num_classes, len(np.unique(labels_true))), random_state=0).fit(latent_rep)
        labels_pred = km.labels_

        purity = metric.compute_purity(labels_pred, labels_true)
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)

        if val:
            data_split = 'Validation'
        else:
            data_split = 'Test'
            #data_split = 'All'

        print('Data = {}, Model = {}, sampler = {}, z_dim = {}, beta_reg = {}'
              .format(self.data, self.model, self.sampler, self.z_dim, self.beta_reg))
        print(' #Points = {}, K = {}, Purity = {},  NMI = {}, ARI = {},  '
              .format(latent_rep.shape[0], self.num_classes, purity, nmi, ari))

        with open('logs/Res_{}_{}.txt'.format(self.data, self.model), 'a+') as f:
                f.write('{}, {} : K = {}, z_dim = {}, beta_reg = {}, sampler = {}, Purity = {}, NMI = {}, ARI = {}\n'
                        .format(timestamp, data_split, self.num_classes, self.z_dim, self.beta_reg,
                                self.sampler, purity, nmi, ari))
                f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='pendigit')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--sampler', type=str, default='one_hot')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--dz', type=int, default=5)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--beta_reg', type=float, default=10.0)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='False')

    args = parser.parse_args()
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)

    num_classes = args.K
    dim_gen = args.dz
    n_cat = 1
    batch_size = args.bs
    beta_reg = args.beta_reg
    timestamp = args.timestamp

    z_dim = dim_gen + num_classes * n_cat
    d_net = model.Discriminator()
    g_net = model.Generator(z_dim=z_dim)
    xs = data.DataSampler()
    zs = util.sample_Z

    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model, args.sampler,
                          num_classes, dim_gen, n_cat, batch_size, beta_reg)

    if args.train == 'True':
        wgan.train()
    else:

        print('Attempting to Restore Model ...')
        if timestamp == '':
            wgan.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            wgan.load(pre_trained=False, timestamp=timestamp)

        wgan.recon_enc(timestamp, val=False)
