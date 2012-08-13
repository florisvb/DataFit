import numpy
np = numpy

import copy

import scipy
import scipy.linalg
from scipy import optimize


import matplotlib.pyplot as plt
import matplotlib.animation as animation

import floris_plot_lib as fpl


class ModelBase(object):
    def __init__(self, parameters):
        self.parameters = parameters
                
    def set_parameters(self, parameter_names, parameter_values):
        for i, name in enumerate(parameter_names):
            self.parameters[name] = parameter_values[i]

    def get_errors(self, data, x):
        print np.sum( (data - self.get_val(x))**2 )
        print data.shape, self.get_val(x).shape
        return data - self.get_val(x)
        
    def fit(self, data, inputs, plot=False, method='optimize'):
    
        # Iterative Optimization Method
        if method == 'optimize':
            print 'Fitting Linear Model with: scipy.optimize.leastsq'
            def f(parameter_values, parameter_names):
                self.set_parameters(parameter_names, parameter_values)
                return self.get_errors(data, inputs)
            
            parameter_values = []
            parameter_names = []
            for name, value in self.parameters.items():
                parameter_values.append(value)
                parameter_names.append(name)
            optimize.leastsq(f, parameter_values, parameter_names)

        # Linear Algebra Method            
        elif method == 'linear':
            x = inputs
            print 'Fitting Linear Model with: scipy.linalg.lstsq'
            n_inputs = 2
            n_outputs = 1
            
            def reshape(a, n_samples):
                return a.reshape(n_samples, 1)
                
            def check_shape(x, n_samples):
                if type(x) is list:
                    x = np.array(x)
                if len(x.shape) == 1:
                    x_ = reshape(x, n_samples)
                elif x.shape[1] > x.shape[0]:
                    x_ = reshape(x, n_samples)
                else:
                    x_ = copy.copy(x)
                return x_
                
            n_samples = np.max(x.shape)
            x_ = check_shape(x, n_samples)
            data_ = check_shape(data, n_samples)
                            
                
            matrix = numpy.hstack( (x_, np.ones([n_samples, 1]), data_) )
            input_columns = range(n_inputs) # the first columns of the array
            output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
        
            A = numpy.vstack([matrix[:,i] for i in input_columns]).T
            B = numpy.vstack([matrix[:,i] for i in output_columns]).T
            params,resids,rank,s = scipy.linalg.lstsq(A,B)
            
            self.parameters['slope'] = params[0,0]
            self.parameters['intercept'] = params[1,0]
            
        else:
            raise ValueError('Method type not recognized, choices are: optimize, linear')
            
    
    def plot(self, inputs):
        if type(inputs) is list:
            deg = len(inputs)
        else:
            deg = 1
        
        if deg == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            vals = self.get_val(inputs)
            ax.plot(inputs, vals, 'r.')
        
        elif deg == 2:
            pass            
            
###############################################################################################################
# Linear Models
###############################################################################################################

class LinearModel(ModelBase):
    def __init__(self, parameters={'slope': 8, 'intercept': 0}):
        self.parameters = parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        return self.parameters['slope']*x + self.parameters['intercept']
        
###############################################################################################################
# Gaussian Models
###############################################################################################################
            
class GaussianModelND(ModelBase):
    def __init__(self, dim=2, parameters=None):
        self.dim = dim
        self.init_parameters(parameters)
        
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {'magnitude': 1}
            for n in range(self.dim):
                mean_key = 'mean_' + str(n)
                std_key = 'std_' + str(n)
                parameters.setdefault(mean_key, 0)
                parameters.setdefault(std_key, 1)
        self.parameters = parameters
                        
    def get_val(self, inputs):
        if type(inputs) is not list:
            inputs = [inputs]
        magnitude = self.parameters['magnitude']
        gauss_terms = []
        for i in range(self.dim):
            mean_key = 'mean_' + str(i)
            std_key = 'std_' + str(i)
            mean = self.parameters[mean_key]
            std = self.parameters[std_key]
            gauss_term = (0.5*(inputs[i] - mean)/std)**2
            gauss_terms.append(gauss_term)
        val = magnitude*np.exp(-1*sum(gauss_terms))
        return val
        
class GaussianModel1D(GaussianModelND):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize'):
        data, edges = np.histogram(x, bins=bins)
        inputs = [np.diff(edges) + edges[0:-1]]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(inputs[0], data, '.')
            ax.plot(inputs[0], self.get_val(inputs), 'r')
            
class GaussianModel2D(GaussianModelND):
    def __init__(self, parameters=None):
        self.dim = 2
        self.init_parameters(parameters)
        
    def fit_occurences(self, pts, bins=40, plot=False, method='optimize'):
        x = pts[0]
        y = pts[1]
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        xedges = np.diff(xedges) + xedges[0:-1]
        yedges = np.diff(yedges) + yedges[0:-1]

        n_samples = np.product(H.shape)
        x, y = np.meshgrid(xedges, yedges)
        y = y.reshape(n_samples)
        x = x.reshape(n_samples)
        data = H.reshape(n_samples)
        inputs = [x, y]
        

        print data.shape
        print x.shape, y.shape
        
        self.fit(data, inputs, plot=plot, method=method)
        
    def get_array_2d(self, xlim, ylim, resolution):
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        inputs = [x,y]
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        return self.get_val(inputs), extent
            
        
######################################################################################
# Gaussian Models -- Time Varying
######################################################################################

class GaussianModelND_TimeVarying(ModelBase):
    def __init__(self, dim=2, parameters=None):
        self.dim = dim
        self.init_parameters(parameters)
        
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {'magnitude': 1}
            for n in range(self.dim):
                mean_intercept_key = 'mean_intercept_' + str(n)
                mean_slope_key = 'mean_slope_' + str(n)
                std_intercept_key = 'std_intercept_' + str(n)
                std_slope_key = 'std_slope_' + str(n)
                keys = [mean_intercept_key, mean_slope_key, std_intercept_key, std_slope_key]
                vals = [0, .1, 1, .5]
                for i, key in enumerate(keys):
                    parameters.setdefault(key, vals[i])
        self.parameters = parameters
                        
    def get_val(self, t, inputs):
        if type(inputs) is not list:
            inputs = [inputs]
        magnitude = self.parameters['magnitude']
        gauss_terms = []
        for n in range(self.dim):
            mean_intercept_key = 'mean_intercept_' + str(n)
            mean_slope_key = 'mean_slope_' + str(n)
            std_intercept_key = 'std_intercept_' + str(n)
            std_slope_key = 'std_slope_' + str(n)
        
            mean = self.parameters[mean_intercept_key] + self.parameters[mean_slope_key]*t
            std = self.parameters[std_intercept_key] + self.parameters[std_slope_key]*t

            gauss_term = (0.5*(inputs[n] - mean)/std)**2
            gauss_terms.append(gauss_term)
        val = magnitude*np.exp(-1*sum(gauss_terms))
        return val
        
class GaussianModel2D_TimeVarying(GaussianModelND_TimeVarying):
    def __init__(self, parameters=None):
        self.dim = 2
        self.init_parameters(parameters)
        
    def get_array_2d(self, t, xlim, ylim, resolution):
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        inputs = [x,y]
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        return self.get_val(t, inputs), extent
        
            
###
    
        
######################################################################################
# Examples
######################################################################################

def example_linearmodel_fit(method='optimize'):
    n_pts = 20
    
    x = np.random.random(n_pts)
    noise = np.random.random(n_pts)
    y = x*10 + noise + np.random.random(1)
    
    linearmodel = LinearModel()
    linearmodel.fit(y, x, plot=True, method=method)
    
    if 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(x,y,'b.')
        
        val = linearmodel.get_val(x)
        ax.plot(x,val,'r')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    
    return linearmodel
    
    
    
###

def example_gaussianmodel1d_fit():
    x = np.random.randn(1000)
    gaussianmodel1d = GaussianModel1D()
    gaussianmodel1d.fit_occurences(x, plot=True)
    return gaussianmodel1d
    
###

def example_gaussianmodel2d_fit():
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 5, 1000)
    
    gm = GaussianModel2D()
    gm.fit_occurences([x, y])
    
    if 1:
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        im, extent = gm.get_array_2d([np.min(x), np.max(x)], [np.min(y), np.max(y)], 0.01)
        ax.imshow(im, extent=extent)
        ax.plot(x,y,'b.')
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
    
    return gm
    
    
###

def example_gaussianmodel2d_timevarying_movie(gm=None):

    if gm is None:
    
        parameters = {  'mean_intercept_0': 0,
                        'mean_slope_0':     .2,
                        'mean_intercept_1': 0.16,
                        'mean_slope_1':     0,
                        'std_intercept_0':  0.2,
                        'std_slope_0':      0.05,
                        'std_intercept_1':  0.05,
                        'std_slope_1':      0.02,
                        'magnitude':        1,
                        }
        gm = GaussianModel2D_TimeVarying(parameters=parameters)
    

    fig = plt.figure()
    anim_params = {'t': 0, 'xlim': [0,1], 'ylim': [0,.33], 't_max': 3., 'dt': 0.05, 'resolution': 0.01}
    
    array, extent = gm.get_array_2d(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'])
    im = plt.imshow( array, cmap=plt.get_cmap('jet'))
    
    def updatefig(*args):
        anim_params['t'] += anim_params['dt']
        if anim_params['t'] > anim_params['t_max']:
            anim_params['t'] = 0
                        
        array, extent = gm.get_array_2d(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'])
                    
        im.set_array(array)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, anim_params, interval=50, blit=True)
    plt.show()
    
    
    
    
