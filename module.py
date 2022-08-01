import random
import numpy as np 
import pandas 
import matplotlib.pyplot as plt 
from copy import deepcopy as c 
from scipy.sparse import diags
import os


def SynthesizeData(p, d=50, n=50, r=5):
    #p, d, n, r = args.p, args.d, args.n, args.r
    fill_num = int(p*d*n)
    U_star = np.random.randn(d,r)
    V_star = np.random.randn(n,r)
    S_star = np.zeros(d*n)
    fill = np.random.normal(0,10,fill_num)
    loc = np.arange(d*n)
    random.shuffle(loc)
    S_star[loc[:fill_num]] = fill
    X = U_star@V_star.T + S_star.reshape(d,n)
    U0 = np.random.randn(d,r)
    V0 = np.random.randn(n,r)
    return X,U_star@V_star.T,U0,V0



class MF_Model(object):
    def __init__(self,args,X):
        self.X = X
        self.model_name = args.model_name
        if self.model_name == 'A_IRLS':
            self.W_U = np.ones([args.n, args.d])
            self.W_V = np.ones([args.d, args.n])
        
    def __call__(self,U,V,y,k=0, target_grad = 'u'):
        if self.model_name == 'subgradient':
            return self.subgradient(U,V,target = target_grad)
        if self.model_name == 'A_IRLS_combined':
            coef = U if target_grad == 'v' else V
            return self.A_IRLS_combined(coef,target = target_grad)
        if self.model_name == 'A_IRLS':
            coef = U if target_grad == 'v' else V
            W = self.W_V if target_grad == 'v' else self.W_U
            return self.A_IRLS(y,coef, k, target_grad)
    
    def subgradient(self,U,V,target = 'u'):
        X = self.X
        res = X-U@V.T
        print('the error is: ', np.sum(abs(res)))
        t = np.ones_like(X)
        t[res < 0] = -1
        t[res == 0] = np.random.uniform(-1+1e-6,1)
        if target == 'v':
            return t.T@U 
        else:
            return t@V

    def A_IRLS_combined(self, coef, cri = 0.01, max_iter = 150,delta = 1e-5,target = 'v'):
        X = self.X
        d,n = X.shape
        _,r = coef.shape
        if target == 'v':
            X_vec = X.transpose(1,0).reshape(-1)
            coef_expand = np.zeros((d*n,r*n))
            for i in range(n):
                coef_expand[d*i:d*(i+1),r*i:r*(i+1)] = coef
        if target == 'u':
            X_vec = X.reshape(-1)
            coef_expand = np.zeros((n*d,r*d))
            for i in range(d):
                coef_expand[n*i:n*(i+1),r*i:r*(i+1)] = coef
        W0 = np.eye(d*n)
        diff = float('inf')
        for _ in range(max_iter):
            if diff <= cri:
                break
            beta = np.linalg.inv(coef_expand.T@W0@coef_expand)@coef_expand.T@W0@X_vec
            tr = np.abs(X_vec-coef_expand@beta)
            tr[np.where(tr < delta)] = delta
            W1 = np.diag(1/tr)
            diff = np.abs(np.sum(W1-W0))/(d*n)
            W0 = c(W1)
        return beta.reshape(d,r) if target == 'u' else beta.reshape(n,r)

    def A_IRLS(self, y, coef, k, target_grad, cri = 0.01, delta = 1e-5):
        n,_ = coef.shape
        coef = coef.astype(np.float32)
 

        W = self.W_V if target_grad == 'v' else self.W_U
        w = diags(W[k]) 
        y = y.astype(np.float32)


        beta = np.linalg.inv(coef.T@w@coef)@coef.T@w@y.astype(np.float32)
        tr = np.abs(y-coef@beta)
        tr[np.where(tr < delta)] = delta
        if target_grad == 'v':
            self.W_V[k] = 1/tr
        else:
            self.W_U[k] = 1/tr
 
        return beta

def plot_error(err_ls, args):
    model_name= args.model_name
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(len(err_ls)),err_ls,LineWidth = 1)
    plt.title(f'Error Curve with {model_name} algorithm')
    plt.xlabel(r'#iter')
    plt.ylabel(r'$\|\|UV^T-L^{\star}\|\|$')
    plt.show()

    
