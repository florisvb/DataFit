import numpy as np
import matplotlib.pyplot as plt


class H_LTI:
    def __init__(self, H):
        '''
        Convenience class for linear time invariant H basis functions
        '''
        self.H = H
        
    def __get_H__(self, t):
        H = [h(t) for h in self.H]
        return H
        
    def __get_H_matrix__(self, tlist):
        if type(tlist) == np.ndarray:
            tlist = tlist.tolist() # make sure it's a list
        if type(tlist) is not list:
            tlist = [tlist]
        H_as_list = [self.__get_H__(t) for t in tlist]
        H = np.matrix(H_as_list)
        return H
        
    def __call__(self, t):
        H = self.__get_H_matrix__(t)
        return H
        
class LeastSquaresEstimator:
    def __init__(self):

        self.xhat = None
        self.Pk = None
        
        self.xhat_history = []
        
    def __reformat_H__(self, H):
        if type(H) is list:
            H = np.matrix(H)
        elif type(H) is np.ndarray:
            H = np.matrix(H)
        return H
        
    def fit(self, y, H, W=None):
        '''
        y       -- list of measurements
        H       -- H matrix, or list of lists
        W       -- list of weights, or matrix of weights, default is identity
        
        Fit data using weighted least squares
        '''
        
        H = self.__reformat_H__(H)
            
        ndim = H.shape[1]
        
        # make sure have enough measurements, make y a matrix
        if len(y) < ndim:
            raise ValueError('Not enough measurements')
        y = np.matrix(y).T
        
        # make sure W is a matrix
        if W is None:
            W = np.matrix(np.eye(len(y)))
        if type(W) is list:
            W = np.matrix(np.diag(W))
         
        xhat = ((H.T*W*H).I) * H.T*W*y
        self.xhat = xhat
        
        return xhat
        
    def initialize_from_data(self, y, H, W=None):
        '''
        Calculates initial xhat based on data, 
        and initial covariance based on the standard deviation of the error
        '''
        
        xhat = self.fit(y, H, W=W)
        errors = self.get_errors(y, H)
        std_err = np.std(errors) # standard deviation of the error between the measurement and the model estimate
        self.Pk = np.matrix(np.eye(len(xhat))*std_err**(-2))
        
        self.xhat_history.append(xhat)
        
    def initialize_with_guess(self, xhat, Pk):
        self.xhat = np.matrix(xhat).T
        self.Pk = Pk
    
    def get_value(self, H):
        H = self.__reformat_H__(H)
        y = H*self.xhat
        return y
    
    def get_errors(self, y, H):
        model_values = self.get_value(H)
        errors = y - model_values
        return errors            
    
    
    def update(self, yk1, Hk1, Wk1=None):  
        '''
        ts  -- list of times measurements were taken
        yk1 -- list of measurements
        Wk1 -- list of weights, or matrix of weights, default is identity
        
        Add a measurement to an existing estimate
        
        xhat: nx1
        yk1 = mx1
        Pk = nxn
        Wk1 = mxm
        
        Kk1: nxm
        Hk1: mxn
        '''
        
        if self.xhat is None:
            raise ValueError('Need to initialize xhat -- see self.initialize_with_guess')
        if self.Pk is None:
            raise ValueError('Need to initialize covariance -- see self.initialize_with_guess')
        
        if type(yk1) is not list:
            yk1 = [yk1]
        
        # make sure W is a matrix
        if Wk1 is None:
            Wk1 = np.matrix(np.eye(len(yk1)))
        if type(Wk1) is list:
            Wk1 = np.matrix(np.diag(Wk1))
                    
        yk1 = np.matrix(yk1).T
        Hk1 = self.__reformat_H__(Hk1)
        
        # grab priors
        Pk = self.Pk
        xhat = self.xhat
        
        # kalman gain
        Kk1 = Pk*Hk1.T*( (Hk1*Pk*Hk1.T + Wk1.I).I )
        
        # covariance
        KH = Kk1*Hk1
        I = np.matrix(np.eye(KH.shape[0]))
        Pk1 = (I-KH)*Pk
        
        # update xhat
        xhatk1 = xhat + Kk1*(yk1 - Hk1*xhat)
        
        # save 
        self.Pk = Pk1
        self.xhat = xhatk1
        
        xhat_as_list = [item for sublist in xhatk1.tolist() for item in sublist]
        self.xhat_history.append(xhat_as_list)
        
    
    
    
    
    
