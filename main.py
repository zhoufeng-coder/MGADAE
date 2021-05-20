import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
from scipy.io import loadmat


def PredictScore(t_mir_dis_matrix, drug_matrix, dis_matrix, seed, epochs, dp, lr,  adjdp):
    np.random.seed(seed)
    #tf.compat.v1.reset_default_graph()
    #tf.compat.v1.set_random_seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(t_mir_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = t_mir_dis_matrix.sum()
    X = constructNet(t_mir_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = t_mir_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, t_mir_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=t_mir_dis_matrix.shape[0], num_v=t_mir_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res



def cross_validation_experiment(mir_dis_matrix, mir_matrix, dis_matrix, seed, epochs, dp, lr, adjdp):
    index_matrix = np.mat(np.where(mir_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 5))
    print("seed=%d, evaluating drug-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        t_matrix = np.matrix(mir_dis_matrix, copy=True)
        t_matrix[tuple(np.array(random_index[k]).T)] = 0
        mir_len = mir_dis_matrix.shape[0]
        dis_len = mir_dis_matrix.shape[1]
        mir_disease_res = PredictScore(
            t_matrix, mir_matrix, dis_matrix, seed, epochs, dp, lr,  adjdp)
        predict_y_proba = mir_disease_res.reshape(mir_len, dis_len)
        metric_tmp = cv_model_evaluate(
            mir_dis_matrix, predict_y_proba, t_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del t_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric


if __name__ == "__main__":



    mir_mkl = np.loadtxt('../data/m_mkl.csv', delimiter=',')
    dis_mkl = np.loadtxt('../data/d_mkl.csv', delimiter=',')
    mir_dis_matrix = np.loadtxt('../data/mda.csv', delimiter=',')


    epoch = 4000
    emb_dim = 128
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    result = np.zeros((1, 5), float)
    average_result = np.zeros((1, 5), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            mir_dis_matrix, mir_mkl, dis_mkl, i, epoch,  dp, lr, adjdp)
    average_result = result / circle_time
    print(average_result)
