from scipy import optimize
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

def fit(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = arange(y.shape[0])
    p = [param() for param in parameters]
    optimize.leastsq(f, p)
    
def linear_fit(x, y, plot=False):
        
    # giving initial parameters
    slope = Parameter(7)
    intercept = Parameter(3)

    # define your function:
    def f(x): return slope() * x + intercept()

    # fit! (given that data is an array with the data to fit)
    fit(f, [slope, intercept], y, x)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(x,y,'.')
        ax.plot(x, x*slope() + intercept(), 'r')
        
        plt.show()
    
    return slope, intercept
