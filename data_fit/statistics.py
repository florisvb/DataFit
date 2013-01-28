import numpy
np = numpy


def get_linear_correlation(x,y):
    '''
    len(x) == len(y)
    return r squared
    '''
    
    N = len(x)
    numerator = (N*np.sum(x*y) - np.sum(x)*np.sum(y))**2
    a = N*np.sum(x**2) - np.sum(x)**2
    b = N*np.sum(y**2) - np.sum(y)**2
    denomenator = (a)*(b)
    rsq = numerator / denomenator
    return rsq
