import numpy
np = numpy

import copy

import scipy
import scipy.linalg
from scipy import optimize

import ransac as ransac

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import floris_plot_lib as fpl


class ModelBase(object):
                
    def set_parameters(self, parameter_names, parameter_values):
        for i, name in enumerate(parameter_names):
            self.parameters[name] = parameter_values[i]

    def get_errors(self, data, inputs=None, parameters=None):
        if parameters is not None:
            self.parameters = parameters
        if inputs is None:
            inputs = [data[:,1:]] #.reshape(1, len(data)) for i in range(1, data.shape[1])
            data = data[:,0]#.reshape(1, len(data))
        
        if len(data.shape) == 2:
            data = data.reshape(len(data))
        
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 2:
                inputs[i] = inp.reshape(len(inp))
        
        ans = data - self.get_val(inputs)
        if len(ans.shape) == 2:
            ans = ans.reshape(np.max(ans.shape))
        return ans
        
    def fit(self, data, inputs=None, plot=False, method='optimize', ignore_parameter_names=[]):
    
        if inputs is None:
            inputs = [data[:,i].reshape(len(data)) for i in range(1, data.shape[1])]
            data = data[:,0].reshape(len(data))
            
        # Iterative Optimization Method
        if method == 'optimize':
            #print 'Fitting Linear Model with: scipy.optimize.leastsq'
            def f(parameter_values, parameter_names):
                self.set_parameters(parameter_names, parameter_values)
                ans = self.get_errors(data, inputs)
                if len(ans.shape) == 2 and ans.shape[0] == 1:
                    ans = ans.reshape(ans.shape[1])
                return ans
            
            parameter_values = []
            parameter_names = []
            for name, value in self.parameters.items():
                if name in ignore_parameter_names:
                    continue
                else:
                    parameter_values.append(value)
                    parameter_names.append(name)
            optimize.leastsq(f, parameter_values, parameter_names)

        # Linear Algebra Method            
        elif method == 'linear':
            x = inputs
            #print 'Fitting Linear Model with: scipy.linalg.lstsq'
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
            
        return self.parameters
            
    def ransac(self, data, inputs=None, parameters=None, min_data_vals=50, max_iterations=10, threshold=2e3, num_vals_req=100):
        if inputs is None:
            inputs = data[1:,:]
            data = data[0,:]
        data = np.vstack((data, inputs)).T
        
        if parameters is None:
            parameters = self.parameters
        ransac_fit = ransac.ransac(data,self,min_data_vals, max_iterations, threshold, num_vals_req,debug=True,return_all=False)
        
        print 'ransac fit: ', ransac_fit
        self.parameters = ransac_fit
        
        return self.parameters
        
    def get_array_2d(self, xlim, ylim, resolution):
        assert(self.dim==2)
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        inputs = [x,y]
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        return self.get_val(inputs), extent
    
    def plot(self, inputs):
        if inputs is not None:
            if self.dim == 1:
                if type(inputs) is list:
                    inputs = inputs[0]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                vals = self.get_val(inputs)
                ax.plot(inputs, vals, 'r.')
            elif deg == 2:
                pass        
                
    def show_fit(self, data=None, inputs=None, ax=None, lims=[], resolution=0.001):
        
        if inputs is not None:
            if type(inputs) is not list: inputs = [inputs]
        if len(lims) != self.dim: lims = None

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        # 1 dimension
        if self.dim == 1:
            if inputs is not None:
                if type(inputs) is list:
                    inputs = inputs[0]
                if lims is None:
                    xlim = [np.min(inputs), np.max(inputs)]
                # plot raw data
                ax.plot(inputs, data, '.') 
            else:
                if lims is None:
                    xlim = [0,1]
            
            # plot fit
            x = np.arange(xlim[0], xlim[-1], resolution)
            vals = self.get_val(x)
            ax.plot(x, vals, linewidth=3)
            
            
            
        # 2 dimensions
        if self.dim == 2:
            if lims is None:
                xlim = [np.min(inputs[0]), np.max(inputs[0])]
                ylim = [np.min(inputs[1]), np.max(inputs[1])]
            
            im, extent = self.get_array_2d(xlim, ylim, resolution)
            ax.imshow(im, extent=extent, origin='lower')
            
            if inputs is not None:
                for i in range(len(inputs[0])):
                    x = inputs[0][i]
                    y = inputs[1][i]
                    ax.plot(x,y,'o', markerfacecolor='none', markeredgewidth=2)
                
            
###############################################################################################################
# Linear Models
###############################################################################################################

class LinearModel(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 1
        if parameters is None:
            self.parameters = {'slope': 8, 'intercept': 0}
        else:
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
            std = np.abs(self.parameters[std_key])
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
            
    def fit_with_guess(self, data, inputs=None, plot=False, method='optimize', ignore_parameter_names=[]):
        if type(inputs) is list:
            inputs = inputs[0]
            
        argsorted = np.argsort(inputs).tolist()
        inputs = inputs[argsorted]
        data = data[argsorted]
        
        argmax = np.argmax(data)
        self.parameters['mean_0'] = inputs[argmax]

        # guess std dev        
        inrange = np.where(inputs > self.parameters['mean_0'] / 4.)[0].tolist()
        std_min = (np.min(inputs[inrange]) - self.parameters['mean_0'])/(2.)
        std_max = (np.max(inputs[inrange]) + self.parameters['mean_0'])/(2.)
        self.parameters['std_0'] = np.mean([std_min, std_max])

        self.parameters['magnitude'] = np.max(data)
        
        return self.fit(data, inputs=inputs, plot=plot, method=method, ignore_parameter_names=ignore_parameter_names)
            
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
        
        self.fit(data, inputs, plot=plot, method=method)
        
    def fit_with_guess(self, data, inputs=None, plot=False, method='optimize'):
        
        argmax = np.argmax(data)
        self.parameters['mean_0'] = inputs[0][argmax]
        self.parameters['mean_1'] = inputs[1][argmax]
        
        # guess std dev        
        inrange = np.where(inputs > self.parameters['mean_0'] / 4)[0].tolist()
        std_min = (np.min(inputs[0][inrange]) - self.parameters['mean_0'])/(-2.)
        std_max = (np.max(inputs[0][inrange]) + self.parameters['mean_0'])/(2.)
        self.parameters['std_0'] = np.mean([std_min, std_max])
        
        inrange = np.where(inputs > self.parameters['mean_1'] / 4)[0].tolist()
        std_min = (np.min(inputs[1][inrange]) - self.parameters['mean_1'])/(-2.)
        std_max = (np.max(inputs[1][inrange]) + self.parameters['mean_1'])/(2.)
        self.parameters['std_1'] = np.mean([std_min, std_max])
        
        self.parameters['magnitude'] = np.max(data)
        
        return self.fit(data, inputs=inputs, plot=plot, method=method)
        
class GaussianModel3D(GaussianModelND):
    def __init__(self, parameters=None):
        self.dim = 3
        self.init_parameters(parameters)
        
    def fit_occurences(self, pts, bins=40, plot=False, method='optimize'):
        x = pts[0]
        y = pts[1]
        z = pts[2]
        pts_arr = np.zeros([len(x), 3])
        print pts_arr.shape, x.shape
        pts_arr[:,0] = x
        pts_arr[:,1] = y
        pts_arr[:,2] = z
    
        H, edges = np.histogramdd(pts_arr, bins=bins)
        xedges = np.diff(edges[0]) + edges[0][0:-1]
        yedges = np.diff(edges[1]) + edges[1][0:-1]
        zedges = np.diff(edges[2]) + edges[2][0:-1]

        n_samples = np.product(H.shape)
        #x, y, z = np.mgrid(xedges, yedges, zedges)
        
        np.mgrid[xedges[0]:xedges[-1]:10j, 0:1:10j, 0:1:10j].shape
        
        y = y.reshape(n_samples)
        x = x.reshape(n_samples)
        z = z.reshape(n_samples)
        data = H.reshape(n_samples)
        inputs = [x, y, z]
        
        self.fit(data, inputs, plot=plot, method=method)
        
    def get_array_2d_slice(self, xlim, ylim, resolution, perpendicular_axis=2, axis_slice=0):
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        z = np.ones_like(x)*axis_slice
        
        if perpendicular_axis == 2:
            inputs = [x,y,z]
        elif perpendicular_axis == 1:
            inputs = [x,z,y]
        elif perpendicular_axis == 0:
            inputs = [z,x,y]
            
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        return self.get_val(inputs), extent
        
    def fit_with_guess(self, data, inputs=None, plot=False, method='optimize'):
        
        argmax = np.argmax(data)
        self.parameters['mean_0'] = inputs[0][argmax]
        self.parameters['mean_1'] = inputs[1][argmax]
        self.parameters['mean_2'] = inputs[2][argmax]
        
        # guess std dev        
        inrange = np.where(inputs > self.parameters['mean_0'] / 4)[0].tolist()
        std_min = (np.min(inputs[0][inrange]) - self.parameters['mean_0'])/(2.)
        std_max = (np.max(inputs[0][inrange]) + self.parameters['mean_0'])/(2.)
        self.parameters['std_0'] = np.mean([std_min, std_max])
        
        inrange = np.where(inputs > self.parameters['mean_1'] / 4)[0].tolist()
        std_min = (np.min(inputs[1][inrange]) - self.parameters['mean_1'])/(2.)
        std_max = (np.max(inputs[1][inrange]) + self.parameters['mean_1'])/(2.)
        self.parameters['std_1'] = np.mean([std_min, std_max])
        
        inrange = np.where(inputs > self.parameters['mean_2'] / 4)[0].tolist()
        std_min = (np.min(inputs[2][inrange]) - self.parameters['mean_2'])/(2.)
        std_max = (np.max(inputs[2][inrange]) + self.parameters['mean_2'])/(2.)
        self.parameters['std_2'] = np.mean([std_min, std_max])
        
        self.parameters['magnitude'] = np.max(data)
        
        return self.fit(data, inputs=inputs, plot=plot, method=method)
        
######################################################################################
# Gaussian Models -- Time Varying
######################################################################################

class GaussianModelND_TimeVarying(ModelBase):
    def __init__(self, dim=2, parameters=None):
        self.dim = dim
        self.init_parameters(parameters)
        
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {'magnitude_intercept': 1, 'magnitude_slope': 0}
            for n in range(self.dim):
                mean_intercept_key = 'mean_' + str(n) + '_intercept'
                mean_slope_key = 'mean_' + str(n) + '_slope'
                std_intercept_key = 'std_' + str(n) + '_intercept'
                std_slope_key = 'std_' + str(n) + '_slope'
                keys = [mean_intercept_key, mean_slope_key, std_intercept_key, std_slope_key]
                vals = [0, .1, 1, .5]
                for i, key in enumerate(keys):
                    parameters.setdefault(key, vals[i])
        self.parameters = parameters
                        
    def get_val(self, inputs):
        if type(inputs) is not list:
            inputs = [inputs]
        gauss_terms = []
        t = inputs[0]
        inputs = inputs[1]
        magnitude = self.parameters['magnitude_intercept'] + self.parameters['magnitude_slope']*t
        for n in range(self.dim):
            mean_intercept_key = 'mean_' + str(n) + '_intercept'
            mean_slope_key = 'mean_' + str(n) + '_slope'
            std_intercept_key = 'std_' + str(n) + '_intercept'
            std_slope_key = 'std_' + str(n) + '_slope'
        
            mean = self.parameters[mean_intercept_key] + self.parameters[mean_slope_key]*t
            std = self.parameters[std_intercept_key] + self.parameters[std_slope_key]*t

            gauss_term = (0.5*(inputs[n] - mean)/std)**2 # note n-1 to account for time variable
            gauss_terms.append(gauss_term)
        val = magnitude*np.exp(-1*sum(gauss_terms))
        return val
        
    def fit(self, timestamps, data, inputs):
        # inputs should not be time varying
        # data should be a list of the data at different timepoints
        
        gm_list = []
        for i, t in enumerate(timestamps):  
            if self.dim == 2:
                gm = GaussianModel2D()
            elif self.dim == 1:
                gm = GaussianModel1D()
            gm.fit_with_guess(data[i], inputs)
            gm_list.append(gm)
            
        parameter_dict = {}
        time_indep_parameter_dict = {}
        for key in self.parameters.keys():
            if '_slope' in key:
                s = key.split('_slope')[0]
                parameter_dict.setdefault(s,np.zeros_like(timestamps)) 
            elif '_slope' not in key and '_intercept' not in key:
                time_indep_parameter_dict.setdefault(key,np.zeros_like(timestamps)) 
            
        for i, gm in enumerate(gm_list):
            for key, parameter in parameter_dict.items():
                parameter[i] = gm.parameters[key]
            for key, parameter in time_indep_parameter_dict.items():
                parameter[i] = gm.parameters[key]
                
        # fix absolute values
        for key, parameter in parameter_dict.items():
            if 'std' in key:
                parameter_dict[key] = np.abs(parameter)
                
        # now make some linear models
        lm_dict = {}
        for key, parameter in parameter_dict.items():
            lm = LinearModel()
            lm.fit(parameter, timestamps)
            lm_dict.setdefault(key, lm)
        
        for key, lm in lm_dict.items():
            key_slope = key + '_slope'
            self.parameters[key_slope] = lm.parameters['slope']
            key_slope = key + '_intercept'
            self.parameters[key_slope] = lm.parameters['intercept']
        
        # time independent parameters
        for key, parameter in time_indep_parameter_dict.items():
            self.parameters[key] = np.median(parameter) 
        
class GaussianModel1D_TimeVarying(GaussianModelND_TimeVarying):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
        
class GaussianModel2D_TimeVarying(GaussianModelND_TimeVarying):
    def __init__(self, parameters=None):
        self.dim = 2
        self.init_parameters(parameters)
        
    def get_array_2d(self, t, xlim, ylim, resolution):
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        inputs = [t,[x,y]]
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        return self.get_val(inputs), extent
        
            
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

def example_linear_ransac(plot=True):
    n_pts = 1000
    n_big_noise_pts = 200

    x = np.random.random(n_pts)
    noise = np.random.random(n_pts)
    y = x*10 + noise + np.random.random(1)
    big_noise = x[0:n_big_noise_pts]*30 + (np.random.random(n_big_noise_pts)-0.5)*20
    y[0:n_big_noise_pts] += big_noise

    linearmodel = LinearModel()
    ransac_fit = linearmodel.ransac(y,x,min_data_vals=50, max_iterations=20, threshold=1e-10, num_vals_req=200)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    linearmodel.show_fit(y, x, ax=ax)
    
    linearmodel.fit(y,x)
    linearmodel.show_fit(ax=ax)
    
    return ransac_fit, linearmodel
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

def example_gaussianmodel3d_fit():
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 5, 1000)
    z = np.random.normal(0, 2, 1000)
    
    gm = GaussianModel3D()
    gm.fit_occurences([x, y, z])
    
    if 0:
    
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
    
        parameters = {  'mean_0_intercept': 0,
                        'mean_0_slope':     .2,
                        'mean_1_intercept': 0.16,
                        'mean_1_slope':     0,
                        'std_0_intercept':  0.2,
                        'std_0_slope':      0.05,
                        'std_1_intercept':  0.05,
                        'std_1_slope':      0.02,
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
    
    
    
    
