"""
K Stochastic Neighbor (kNCA)
Ported to Python from https://github.com/kswersky/k-stochastic-neighbor
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange

import sys
from scipy.optimize import fmin_l_bfgs_b as lbfgs
import time

from ctypes import *
import random

try:
    import matplotlib.pylab as plt
except:
    pass

import pickle
from scipy.spatial.distance import pdist, squareform, cdist

from .base_metric import BaseMetricLearner

np_double_type = np.double
c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

import os
cwd = os.getcwd()
print cwd

KNCALIB__ = np.ctypeslib.load_library("knca_alg.A.dylib", ".")


class kNCA(BaseMetricLearner):
  def __init__(self, k, kmin=1, num_dims=None, max_iter=100, learning_rate=0.1, momentum=0, batch_size=10):
    self.params = {
      'k': k,
      'kmin': kmin,
      'num_dims': num_dims,#P
      'max_iter': max_iter,#num_iters
      'learning_rate': learning_rate,#eta
      'momentum': momentum,
      'batch_size': batch_size
    }
    self.A = None

  def transformer(self):
    return self.A

  def fit(self, X, labels):
    N = X.shape[0]   # num points
    D = X.shape[1]   # input dimension
    C = int(labels.max()+1)    # num classes
    
    k = self.params['k']
    kmin = self.params['kmin']
    batch_size = self.params['bach_size']
    max_iter = self.params['max_iter']
    learning_rate = self.params['learning_rate']
    num_dims = self.params['num_dims']
    momentum = self.params['momentum']

    if num_dims == None:
        num_dims = X.shape[1] #total

    num_batches = np.ceil(np.double(N)/batch_size)
    
    A = self.A

    if (A is None):
        A = np.random.randn(num_dims,D)
        [U,S,V] = np.linalg.svd(A,full_matrices=False)
        S = np.ones(S.shape)
        A = np.dot(U,np.dot(np.diag(S),V))
    else:
        A = A.copy()

    #Makes some objects to be used for performing inference.
    Ccounts = np.array([np.sum(labels == c) for c in range(C)])
    kncas,knca0 = make_kncas(k,kmin,N,Ccounts,C,batch_size)

    #Run SGD
    dA = 0
    for i in range(max_iter):
        randIndices = np.random.permutation(X.shape[0])
        f_tot = 0
        for batch in range(int(num_batches)):
            ind = randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),X.shape[0])]
            f,g = knca(A,X,labels,num_dims,kncas,knca0,ind=ind)
            f_tot += f
            dA = momentum*dA - learning_rate*g
            A += dA

    self.X = X
    self.A = A

    return self

  def knca(self, A,X,y,P,kncas,knca0,ind=None):
    N = X.shape[0]   # num points
    D = X.shape[1]   # input dimension
    C = int(max(y)+1)    # num classes

    sorted_ind = np.argsort(y)
    X = X[sorted_ind,:]
    y = y[sorted_ind]

    if (ind is None):
        ind = range(N)
    else:
        ind = sorted_ind[ind] 

    dA = np.zeros(A.shape)

    Xp = np.dot(X, A.T)   # NxP  low dim projections
    dists = cdist(Xp[ind,:], Xp, 'sqeuclidean')
    dists = np.exp(-dists)

    for (i,ii) in enumerate(ind):
        dists[i,ii] = 0

    y = y[ind].astype(np.double)

    node_marg0_all, log_Z0 = knca0.infer(dists,y)

    dthetas = np.zeros((len(ind),N))
    Z1 = 0
    for knca in kncas:
        node_marg1_all, log_Z1Kp = knca.infer(dists,y)
        if (any(np.isnan(log_Z1Kp))):
            pickle.dump({'logZ':log_Z1Kp,'dists':dists},open('diagnostics','wb'))
            raise Exception('Numerical error in knca training.')
        Z1exp = np.exp(log_Z1Kp)
        node_marg1_all[np.isnan(node_marg1_all)] = 0
        dthetas += node_marg1_all*Z1exp[:,None]
        Z1 += Z1exp

    gains = np.log(Z1) - log_Z0

    dthetas = (dthetas.T / Z1).T
    dthetas -= node_marg0_all

    dT = 0
    for (i,ii) in enumerate(ind):
        Xdiff = (X[ii,:] - X)
        dT -= np.dot(Xdiff.T,dthetas[i,:][:,None]*Xdiff)
    dA = 2*np.dot(A,dT)

    f = -np.sum(gains)/len(ind)
    g = -dA / len(ind)

    return f,g
def make_kncas(self, k,kmin,N,Ccounts,C,batch_size):
    Ccounts = Ccounts.astype(np.double)
    kncas = []
    if (kmin != k):
        for k in range(kmin,int(k+1)/2):
            knca = KNCAAlg(batch_size, N, k, k, Ccounts, C)
            kncas.append(knca)
        knca = KNCAAlg(batch_size, N, k, -2, Ccounts, C)
        kncas.append(knca)
    else:
        knca = KNCAAlg(batch_size, N, k, k, Ccounts, C)
        kncas.append(knca)

    knca0 = KNCAAlg(batch_size, N, k, -1, Ccounts, C)

    return kncas,knca0

class KNCAAlg:

    def __init__(self, batch_size, N, K, Kp, class_counts, C):
        self.N = N
        self.B = batch_size
        self.K = K
        self.Kp = Kp
        self.class_counts = class_counts
        self.C = C

        self.Infer = KNCALIB__["infer"]

        # reuse storage, so we don't keep re-allocating memory
        self.fmsgs1_c = (self.N * (self.K+1) * c_double)()
        self.bmsgs1_c = (self.N * (self.K+1) * c_double)()

        self.fmsgs2_c = ((self.C+1) * (self.K+1) * c_double)()
        self.bmsgs2_c = ((self.C+1) * (self.K+1) * c_double)()

        self.class_counts_c = class_counts.ctypes.data_as(c_double_p)

        self.result_marginals_c = (self.N * self.B * c_double)()
        self.result_log_Zs_c = (self.B * c_double)()


    def infer(self, exp_node_pots, Ys):
        exp_node_pots = (exp_node_pots[:]).astype(np_double_type)
        exp_node_pots_c = exp_node_pots.ctypes.data_as(c_double_p)

        # was having trouble passing this to c as an int.  just using
        # doubles to save some pointless debugging.
        Ys = Ys.astype(np.double)
        Ys_c = Ys.ctypes.data_as(c_double_p)

        #print "Class Counts"
        #print np.ctypeslib.as_array(self.class_counts_c)

        self.Infer(c_int(self.N), c_int(self.B), c_int(self.K), c_int(self.Kp),
                   self.class_counts_c, c_int(self.C), exp_node_pots_c, Ys_c,
                   self.fmsgs1_c, self.bmsgs1_c, self.fmsgs2_c, self.bmsgs2_c,
                   self.result_marginals_c, self.result_log_Zs_c)

        marginals = np.ctypeslib.as_array(self.result_marginals_c).reshape((self.B,self.N))
        log_Zs = np.ctypeslib.as_array(self.result_log_Zs_c).reshape((self.B))

        return marginals, log_Zs
