from tqdm import tqdm_notebook
import numpy as np

class Trainer(object):
    def __init__(self, model, args, X, L_star = None):
        self.model = model
        self.args = args
        self.L_star = L_star
        self.L_star = L_star
        self.error_ls = []
        self.X = X
        
        if args.model_name == 'subgradient':
            self.train = self.train_subgradient
        if args.model_name == 'A_IRLS_combined':
            self.train = self.train_A_ILRS_combined
        if args.model_name == 'A_IRLS':
            self.train = self.train_A_ILRS
    def train_subgradient(self,args,U,V):
        U0,V0 = U,V
        max_iter = args.max_iter_subG
        lr = args.lr
        for _ in tqdm_notebook(range(max_iter)):
            U0 += self.model(U0,V0,self.X, target_grad='u')*lr
            V0 += self.model(U0,V0,self.X, target_grad='v')*lr
            error=np.sum(abs(U0@V0.T-self.X))
            self.error_ls.append(error)
        return U0,V0
    def train_A_ILRS_combined(self,args,U,V):
        U0,V0 = U,V
        max_iter = args.max_iter_A_ILRS_combined
        for _ in tqdm_notebook(range(max_iter)):
            U0 = self.model(U0,V0,self.X, target_grad='u')
            V0 = self.model(U0,V0,self.X, target_grad='v')
            if self.L_star is not None:
                error = np.sum(np.abs(U0@V0.T-self.L_star))
                self.error_ls.append(error)
        return U0,V0
    def train_A_ILRS(self,args,U,V):
        U0,V0 = U,V
        d,n = self.X.shape
        max_iter = args.max_iter_A_ILRS
        for _ in tqdm_notebook(range(max_iter)):
            for k in range(n):
                V0[k,:] = self.model(U0,V0,self.X[:,k], k,target_grad='v')
            for k in range(d):
                U0[k,:] = self.model(U0,V0,self.X[k,:], k, target_grad='u')
            
            if self.L_star is not None:
                error = np.sum(np.abs(U0@V0.T-self.L_star))
                self.error_ls.append(error)
            else:
                error=np.sum(np.abs(self.X - U0@V0.T))
                print('the error is: ', error)
                self.error_ls.append(error)
        return U0,V0



