import numpy
np = numpy

import copy

import scipy
import scipy.linalg
import scipy.special
from scipy import optimize

import ransac as ransac

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import fly_plot_lib.plot as fpl


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
        
        self.data = data
        self.inputs = inputs
        
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
        
        #print 'ransac fit: ', ransac_fit
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
                
    def show_fit(self, data=None, inputs=None, ax=None, lims=[], resolution=0.001, colornorm=None, colormap='jet', axis_slice=0, axis=2, n_inputs_to_show=50, axis_slice_threshold=0.01):
        
        print lims

        if data is None:
            data = self.data
        if inputs is None:
            inputs = self.inputs
        
        if inputs is not None:
            if type(inputs) is not list and type(inputs) != np.ndarray: inputs = [inputs]
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
            else:
                xlim = lims[0]
                ylim = lims[1]
                
            im, extent = self.get_array_2d(xlim, ylim, resolution)
            
            if colornorm is None:
                norm = matplotlib.colors.Normalize(np.min(im), np.max(im))
            else:
                norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
            cmap = matplotlib.cm.ScalarMappable(norm, colormap)

            ax.imshow(im, extent=extent, origin='lower', norm=norm, cmap=colormap)
            
            if inputs is not None and data is not None:
                inputs_to_show = np.linspace(0, len(inputs[0])-1, n_inputs_to_show)
                inputs_to_show = np.array(inputs_to_show, dtype=int)
                for i in inputs_to_show:
                    x = inputs[0][i]
                    y = inputs[1][i]
                    val = data[i]
                    color = cmap.to_rgba(val)
                    ax.plot(x,y,'o', markerfacecolor=color, markeredgewidth=2, markersize=8)
                
        # 3 dimensions
        if self.dim == 3:
            if lims is None:
                xlim = [np.min(inputs[0]), np.max(inputs[0])]
                ylim = [np.min(inputs[1]), np.max(inputs[1])]
                zlim = [np.min(inputs[2]), np.max(inputs[2])]
            else:
                xlim = lims[0]
                ylim = lims[1]
                zlim = lims[2]
                
            print xlim, ylim
            im, extent = self.get_array_2d_slice(xlim, ylim, zlim, resolution, axis=axis, axis_slice=axis_slice)
            
            if colornorm is None:
                norm = matplotlib.colors.Normalize(np.min(im), np.max(im))
            else:
                norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1], clip=True)

            ax.imshow(im, extent=extent, origin='lower', norm=norm, cmap=colormap)
            
            if axis==0:
                ax.set_xlim(ylim)
                ax.set_ylim(zlim)
                ax.set_aspect('equal')
            elif axis==1:
                ax.set_xlim(xlim)
                ax.set_ylim(zlim)
                ax.set_aspect('equal')
            elif axis==2:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect('equal')
            '''
            if inputs is not None and data is not None:
                
                # find inputs that are close to slice:
                inputs_to_show = []
                for i in range(len(inputs[0])):
                    if np.abs(inputs[axis][i] - axis_slice) <= axis_slice_threshold:
                        inputs_to_show.append(i)
                np.random.shuffle(inputs_to_show)
                n = 0
                for i in inputs_to_show:
                    if n > n_inputs_to_show:
                        break
                    else:
                        n += 1
                    if axis == 2:
                        x = inputs[0][i]
                        y = inputs[1][i]
                    if axis == 1:
                        x = inputs[0][i]
                        y = inputs[2][i]
                    if axis == 0:
                        x = inputs[1][i]
                        y = inputs[2][i]
                    val = data[i]
                    color = cmap.to_rgba(val)
                    ax.plot(x,y,'o', markerfacecolor=color, markeredgewidth=2, markersize=8)
            '''
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
# Circle Model
###############################################################################################################

class CircleModel(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 2
        self.init_parameters(parameters)
            
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('center_x',1)
            parameters.setdefault('center_y',1)
            parameters.setdefault('radius',1)
        self.parameters = parameters
                
    def get_val(self, inputs):
        if type(inputs) is list:
            x = np.array(inputs[0])
            y = np.array(inputs[1])
        
        center_x = self.parameters['center_x']
        center_y = self.parameters['center_y']
        radius = self.parameters['radius']
        val = (center_x-x)**2 + (center_y-y)**2 - radius**2
        
        return val


###############################################################################################################
# Exponential Models
###############################################################################################################

class SigmoidModel(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
            
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('tao',1)
            parameters.setdefault('gain',1)
            parameters.setdefault('xshift',0)
            parameters.setdefault('yshift',0)
        self.parameters = parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        tao = self.parameters['tao']
        xshift = self.parameters['xshift']
        yshift = self.parameters['yshift']
        gain = self.parameters['gain']
        val = gain*(1+np.exp(-1*np.abs(tao)*(x+xshift)))**(-1)+yshift
        
        return val


class ExponentialDecay(ModelBase):
    def __init__(self, parameters=None, fixed_parameters={'exp_base': np.exp(1)}):
        self.dim = 1
        self.init_parameters(parameters, fixed_parameters)
            
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('assymptote',1)
            parameters.setdefault('gain',1)
            parameters.setdefault('xoffset',0)
            parameters.setdefault('yoffset',0)
            parameters.setdefault('I',0)
            parameters.setdefault('S',1)

        if fixed_parameters is not None:
            for parameter, val in fixed_parameters.items():
                try:
                    del(parameters[parameter])
                except:
                    pass
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        parameters = {}
        if self.fixed_parameters is not None:
            for p, val in self.fixed_parameters.items():
                parameters[p] = val 
        for p, val in self.parameters.items():
            if p not in parameters.keys():
                parameters[p] = val

        assymptote = parameters['assymptote']
        gain = parameters['gain']
        xoffset = parameters['xoffset']
        yoffset = parameters['yoffset']
        I = parameters['I']
        S = parameters['S']
        exp_base = parameters['exp_base']

        val = assymptote*(I-S*exp_base**(-1*(x-xoffset)*gain)) + yoffset
        
        return val

class PowerCurve(ModelBase):
    def __init__(self, parameters=None, fixed_parameters=None):
        self.dim = 1
        self.init_parameters(parameters, fixed_parameters)
            
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('gain',1)
            parameters.setdefault('xoffset',0)
            parameters.setdefault('yoffset',0)
            parameters.setdefault('k',0)

        if fixed_parameters is not None:
            for parameter, val in fixed_parameters.items():
                del(parameters[parameter])
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        parameters = {}
        if self.fixed_parameters is not None:
            for p, val in self.fixed_parameters.items():
                parameters[p] = val 
        for p, val in self.parameters.items():
            if p not in parameters.keys():
                parameters[p] = val

        gain = parameters['gain']
        xoffset = parameters['xoffset']
        yoffset = parameters['yoffset']
        k = parameters['k']
        
        val = gain*(x-xoffset)**(k)+yoffset
        
        return val
        

class LogDecay(ModelBase):
    def __init__(self, parameters=None, fixed_parameters=None):
        self.dim = 1
        self.init_parameters(parameters, fixed_parameters)
            
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('a',1)
            parameters.setdefault('b',1)
            parameters.setdefault('xoffset',0)
            parameters.setdefault('yoffset',0)
            #fixed_parameters = {}
            #fixed_parameters.setdefault('b', 1)
        if fixed_parameters is not None:
            for parameter, val in fixed_parameters.items():
                del(parameters[parameter])
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        a = self.parameters['a']
        if self.fixed_parameters is not None:
            if 'b' in self.fixed_parameters: 
                b = self.fixed_parameters['b']
        else:
            b = self.parameters['b']
        xoffset = self.parameters['xoffset']
        yoffset = self.parameters['yoffset']
        val = a*np.log( b*(x-xoffset)+1) + yoffset
        
        return val
        

class BoundedExponential(ModelBase):
    def __init__(self, parameters=None, fixed_parameters=None):
        self.dim = 4
        self.init_parameters(parameters, fixed_parameters)
            
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('A',10)
            parameters.setdefault('S',100)
            parameters.setdefault('k',1/100000.)
            parameters.setdefault('I',1000)
            #fixed_parameters = {}
            #fixed_parameters.setdefault('b', 1)
        if fixed_parameters is not None:
            for parameter, val in fixed_parameters.items():
                del(parameters[parameter])
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        A = self.parameters['A']
        S = self.parameters['S']
        k = self.parameters['k']
        I = self.parameters['I']

        val = A*(I-S*np.exp(-k*x))
        
        return val
        
###############################################################################################################
# CDF Model
###############################################################################################################

class CDFModel(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
            
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('mean',0.5)
            parameters.setdefault('std',1)
        self.parameters = parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        mean = self.parameters['mean']
        std = self.parameters['std']
        val = 0.5*(1+scipy.special.erf( (x-mean) / np.sqrt(2*std**2) ) )
        
        return val
        

###############################################################################################################
# Root Models
###############################################################################################################

class RootModel(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
            
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {}
            parameters.setdefault('root',1)
            parameters.setdefault('gain',1)
            parameters.setdefault('xshift',0)
            parameters.setdefault('yshift',0)
        self.parameters = parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        
        root = self.parameters['root']
        xshift = self.parameters['xshift']
        yshift = self.parameters['yshift']
        gain = self.parameters['gain']
        val = gain*(x+xshift)**(root)+yshift
        
        return val
        
                
###############################################################################################################
# Gaussian Models
###############################################################################################################

def get_ndim_gaussian_model(n):
    if n == 1:
        return GaussianModel1D()
    if n == 2:
        return GaussianModel2D()
    if n == 3:
        return GaussianModel3D()
            
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
            gauss_term = 0.5*((inputs[i] - mean)/std)**2
            gauss_terms.append(gauss_term)
        val = magnitude*np.exp(-1*sum(gauss_terms))
        return val
        
class GaussianModel1D(GaussianModelND):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
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
        self.parameters['std_0'] = np.mean([np.abs(std_min), np.abs(std_max)])

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
        std_min = (np.min(inputs[0][inrange]) - self.parameters['mean_0'])/(2.)
        std_max = (np.max(inputs[0][inrange]) + self.parameters['mean_0'])/(2.)
        self.parameters['std_0'] = np.mean([np.abs(std_min), np.abs(std_max)])
        
        inrange = np.where(inputs > self.parameters['mean_1'] / 4)[0].tolist()
        std_min = (np.min(inputs[1][inrange]) - self.parameters['mean_1'])/(2.)
        std_max = (np.max(inputs[1][inrange]) + self.parameters['mean_1'])/(2.)
        self.parameters['std_1'] = np.mean([np.abs(std_min), np.abs(std_max)])
        
        self.parameters['magnitude'] = np.max(data)

        print self.parameters
        
        return self.fit(data, inputs=inputs, plot=plot, method=method)
        
class GaussianModel3D(GaussianModelND):
    def __init__(self, parameters=None):
        self.dim = 3
        self.inputs = None
        self.data = None
        self.init_parameters(parameters)
        
    def fit_occurences(self, pts, bins=40, plot=False, method='optimize'):
        x = pts[0]
        y = pts[1]
        z = pts[2]
        pts_arr = np.zeros([len(x), 3])
        pts_arr[:,0] = x
        pts_arr[:,1] = y
        pts_arr[:,2] = z
    
        H, edges = np.histogramdd(pts_arr, bins=bins)
        xedges = np.diff(edges[0]) + edges[0][0:-1]
        yedges = np.diff(edges[1]) + edges[1][0:-1]
        zedges = np.diff(edges[2]) + edges[2][0:-1]

        n_samples = np.product(H.shape)
        #x, y, z = np.mgrid(xedges, yedges, zedges)
        
        nedges = len(xedges)
        s = str(nedges) + 'j'
        nj = complex(s)
        x,y,z = np.mgrid[xedges[0]:xedges[-1]:nj, yedges[0]:yedges[-1]:nj, zedges[0]:zedges[-1]:nj]
        
        y = y.reshape(n_samples)
        x = x.reshape(n_samples)
        z = z.reshape(n_samples)
        data = H.reshape(n_samples)
        inputs = [x, y, z]
        
        self.fit_with_guess(data, inputs, plot=plot, method=method)
        
    def get_array_2d_slice(self, xlim, ylim, resolution, axis=2, axis_slice=0):
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        z = np.ones_like(x)*axis_slice
        
        if axis == 2:
            inputs = [x,y,z]
        elif axis == 1:
            inputs = [x,z,y]
        elif axis == 0:
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
        self.parameters['std_0'] = np.mean([np.abs(std_min), np.abs(std_max)])
        
        inrange = np.where(inputs > self.parameters['mean_1'] / 4)[0].tolist()
        std_min = (np.min(inputs[1][inrange]) - self.parameters['mean_1'])/(2.)
        std_max = (np.max(inputs[1][inrange]) + self.parameters['mean_1'])/(2.)
        self.parameters['std_1'] = np.mean([np.abs(std_min), np.abs(std_max)])
        
        inrange = np.where(inputs > self.parameters['mean_2'] / 4)[0].tolist()
        std_min = (np.min(inputs[2][inrange]) - self.parameters['mean_2'])/(2.)
        std_max = (np.max(inputs[2][inrange]) + self.parameters['mean_2'])/(2.)
        self.parameters['std_2'] = np.mean([np.abs(std_min), np.abs(std_max)])
        
        self.parameters['magnitude'] = np.max(data)
        
        #print 'Guess: '
        #print self.parameters
        
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
        
    def fit(self, timestamps, data, inputs, return_gm_list_only=False):
        # inputs should not be time varying
        # data should be a list of the data at different timepoints
        
        self.inputs = inputs
        self.data = data
        self.timestamps = timestamps
        
        gm_list = []
        for i, t in enumerate(timestamps):  
            if self.dim == 1:
                gm = GaussianModel1D()
            elif self.dim == 2:
                gm = GaussianModel2D()
            elif self.dim == 3:
                gm = GaussianModel3D()
            gm.fit_with_guess(data[i], inputs)
            gm_list.append(gm)
            
        if return_gm_list_only:
            return gm_list
            
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
            
    def show_fit(self, t, data=None, inputs=None, ax=None, lims=[], resolution=0.001, colornorm=None, colormap='jet', axis_slice=0, axis=2, n_inputs_to_show=50, axis_slice_threshold=0.01, animate=False, timerange=[0,1], dt=0.05):
        
        if data is None:
            data = self.data
        if inputs is None:
            inputs = self.inputs
        
        if inputs is not None:
            if type(inputs) is not list: inputs = [inputs]
        if len(lims) != self.dim: lims = None

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        data_index = np.argmin(np.abs(self.timestamps-t))
            
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
            vals = self.get_val([t, x])
            ax.plot(x, vals, linewidth=3)
            
            
            
        # 2 dimensions
        if self.dim == 2:
            if lims is None:
                xlim = [np.min(inputs[0]), np.max(inputs[0])]
                ylim = [np.min(inputs[1]), np.max(inputs[1])]
            else:
                xlim = lims[0]
                ylim = lims[1]
                
            im, extent = self.get_array_2d(t, xlim, ylim, resolution)
            
            if colornorm is None:
                norm = matplotlib.colors.Normalize(np.min(im), np.max(im))
            else:
                norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
            cmap = matplotlib.cm.ScalarMappable(norm, colormap)

            ax.imshow(im, extent=extent, origin='lower', norm=norm, cmap=colormap)
            
            if inputs is not None and data is not None:
                inputs_to_show = np.linspace(0, len(inputs[0])-1, n_inputs_to_show)
                inputs_to_show = np.array(inputs_to_show, dtype=int)
                for i in inputs_to_show:
                    x = inputs[0][i]
                    y = inputs[1][i]
                    val = data[i]
                    color = cmap.to_rgba(val)
                    ax.plot(x,y,'o', markerfacecolor=color, markeredgewidth=2, markersize=8)
                
        # 3 dimensions
        if self.dim == 3:
            if lims is None:
                if axis == 2:
                    xlim = [np.min(inputs[0]), np.max(inputs[0])]
                    ylim = [np.min(inputs[1]), np.max(inputs[1])]
                elif axis == 1:
                    xlim = [np.min(inputs[0]), np.max(inputs[0])]
                    ylim = [np.min(inputs[2]), np.max(inputs[2])]
                elif axis == 0:
                    xlim = [np.min(inputs[1]), np.max(inputs[1])]
                    ylim = [np.min(inputs[2]), np.max(inputs[2])]
            else:
                xlim = lims[0]
                ylim = lims[1]
                
            im, extent = self.get_array_2d_slice(t, xlim, ylim, resolution, axis=axis, axis_slice=axis_slice)
            
            if colornorm is None:
                norm = matplotlib.colors.Normalize(np.min(im), np.max(im))
            else:
                norm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
            cmap = matplotlib.cm.ScalarMappable(norm, colormap)

            im = ax.imshow(im, extent=extent, origin='lower', norm=norm, cmap=colormap)
            
            if inputs is not None and data is not None:
                
                # find inputs that are close to slice:
                inputs_to_show = []
                for i in range(len(inputs[0])):
                    if np.abs(inputs[axis][i] - axis_slice) <= axis_slice_threshold:
                        inputs_to_show.append(i)
                np.random.shuffle(inputs_to_show)
                n = 0
                for i in inputs_to_show:
                    if n > n_inputs_to_show:
                        break
                    else:
                        n += 1
                    if axis == 2:
                        x = inputs[0][i]
                        y = inputs[1][i]
                    if axis == 1:
                        x = inputs[0][i]
                        y = inputs[2][i]
                    if axis == 0:
                        x = inputs[1][i]
                        y = inputs[2][i]
                    val = data[data_index][i]
                    color = cmap.to_rgba(val)
                    ax.plot(x,y,'o', markerfacecolor=color, markeredgewidth=2, markersize=8, zorder=100)
                    
            if animate:
                anim_params = {'t': timerange[0], 'xlim': xlim, 'ylim': ylim, 't_min': timerange[0], 't_max': timerange[-1], 'dt': dt, 'resolution': resolution}
                def updatefig(*args):
                    anim_params['t'] += anim_params['dt']
                    if anim_params['t'] > anim_params['t_max']:
                        anim_params['t'] = anim_params['t_min']
                                    
                    array, extent = self.get_array_2d_slice(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'], axis=axis, axis_slice=axis_slice)
                    
                    im.set_array(array)
                    return im,
                
                animate_plot = animation.FuncAnimation(fig, updatefig, anim_params, interval=50, blit=True)
                plt.show()
                
    def get_time_trace_at_point(self, timerange, position, resolution=0.01):
        timestamps = np.arange(timerange[0], timerange[1], resolution)
        values = np.zeros_like(timestamps)
        for i, t in enumerate(timestamps):
            values[i] = self.get_val([t, position])
        return timestamps, values
        
    def plot_time_trace_at_point(self, timerange, position, resolution=0.01, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            timestamps, values = self.get_time_trace_at_point(timerange, position, resolution)
            ax.plot(timestamps, values) 
            
    def get_gaussian_model_at_time_t(self, t):
        gm = get_ndim_gaussian_model(self.dim)
        for param in gm.parameters.keys():
            # find slope and intercept key in timevarying gaussian, and get val at time t
            intercept = self.parameters[param+'_intercept']
            slope = self.parameters[param+'_slope']
            gm.parameters[param] = intercept + slope*t
        return gm
        
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
        
class GaussianModel3D_TimeVarying(GaussianModelND_TimeVarying):
    def __init__(self, parameters=None):
        self.dim = 3
        self.init_parameters(parameters)
        
    def get_array_2d_slice(self, t, xlim, ylim, resolution, axis=2, axis_slice=0):
        x = np.arange(xlim[0], xlim[1], resolution, np.float32)
        y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
        z = np.ones_like(x)*axis_slice
        
        if axis == 2:
            inputs = [x,y,z]
        elif axis == 1:
            inputs = [x,z,y]
        elif axis == 0:
            inputs = [z,x,y]
            
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        return self.get_val([t, inputs]), extent
        
###

###############################################################################################################
# Von Mises Models
###############################################################################################################

class VonMisesModel1D(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 1
        if parameters is None:
            self.parameters = {'location': 1, 'concentration': 1}
        else:
            self.parameters = parameters
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
            
        k = self.parameters['concentration']
        u = self.parameters['location']
        bessel = scipy.special.j0(k)
        val = np.exp( k*np.cos(x-u) ) / (2*np.pi*bessel)
            
        return val
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = [np.diff(edges) + edges[0:-1]]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(inputs[0], data, '.')
            ax.plot(inputs[0], self.get_val(inputs), 'r')
            
    def get_integral(self, resolution=1000):
        x = np.linspace(-np.pi,np.pi,resolution)
        y = self.get_val(x)
        return np.abs(np.sum(y)/float(resolution)*np.pi*2)
            
            
######################################################################################
# Sum of Von Mises
######################################################################################

class Sumof1DVonMisesModel(ModelBase):
    def __init__(self, num_vonmises=2, parameters=None):
        self.num_vonmises = num_vonmises
        self.init_parameters(parameters)
        
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {}
            for n in range(self.num_vonmises):
                magnitude_key = 'magnitude_' + str(n)
                location_key = 'location_' + str(n)
                concentration_key = 'concentration_' + str(n)
                parameters.setdefault(location_key, 0)
                parameters.setdefault(concentration_key, 1)
                parameters.setdefault(magnitude_key, 1)
            parameters.setdefault('baseline', .5)
        self.parameters = parameters
        
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        vonmises_terms = []
        for i in range(self.num_vonmises):
            magnitude_key = 'magnitude_' + str(i)
            location_key = 'location_' + str(i)
            concentration_key = 'concentration_' + str(i)
            u = self.parameters[location_key]
            k = np.abs(self.parameters[concentration_key])
            magnitude = np.abs(self.parameters[magnitude_key])
            bessel = np.abs(scipy.special.j0(k))
            val = np.exp( k*np.cos(x-u) ) / (2*np.pi*bessel)
            vonmises_terms.append(magnitude*val)
        vonmises_terms.append(np.ones_like(x)*np.abs(self.parameters['baseline']))
        val = np.vstack(vonmises_terms).sum(axis=0)
        return val
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = np.diff(edges) + edges[0:-1]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            print len(inputs), len(self.get_val(inputs))
            ax.plot(inputs, data, '.')
            ax.plot(np.array(inputs), self.get_val(inputs), 'r')
            
###############################################################################################################
# Von Mises Models with fixed location
###############################################################################################################

class VonMisesModel1DFixedPos(ModelBase):
    def __init__(self, parameters=None):
        self.dim = 1
        if parameters is None:
            self.parameters = {'concentration': 1}
        else:
            self.parameters = parameters
        self.fixed_parameters = {'location': 1}
                
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
            
        k = self.parameters['concentration']
        u = self.fixed_parameters['location']
        bessel = scipy.special.j0(k)
        val = np.exp( k*np.cos(x-u) ) / (2*np.pi*bessel)
            
        return val
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = [np.diff(edges) + edges[0:-1]]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(inputs[0], data, '.')
            ax.plot(inputs[0], self.get_val(inputs), 'r')
            
######################################################################################
# Sum of 4 Von Mises with fixed locations
######################################################################################

class Sumof4x1DVonMisesModelFixedPos(ModelBase):
    def __init__(self, num_vonmises=2, parameters=None, fixed_parameters=None):
        self.num_vonmises = num_vonmises
        self.init_parameters(parameters, fixed_parameters)
        
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {}
            fixed_parameters = {}
            for n in range(self.num_vonmises):
                magnitude_key = 'magnitude_' + str(n)
                location_key = 'location_' + str(n)
                concentration_key = 'concentration_' + str(n)
                fixed_parameters.setdefault(location_key, 0)
                parameters.setdefault(concentration_key, 1)
                parameters.setdefault(magnitude_key, 1)
            fixed_parameters.setdefault('baseline', .5)
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
        
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        vonmises_terms = []
        for i in range(self.num_vonmises):
        
            magnitude_key = 'magnitude_' + str(i)
            location_key = 'location_' + str(i)
            concentration_key = 'concentration_' + str(i)
            u = self.fixed_parameters[location_key]
            
            try:
                k = np.abs(self.fixed_parameters[concentration_key])
            except:  
                k = np.abs(self.parameters[concentration_key])
            try:
                magnitude = np.abs(self.parameters[magnitude_key])
            except:
                magnitude = np.abs(self.parameters[magnitude_key])
        
            bessel = np.abs(scipy.special.j0(k))
            val = np.exp( k*np.cos(x-u) ) / (2*np.pi*bessel)
            vonmises_terms.append(magnitude*val)
            
            if i==1:
                location_key = 'location_' + str(i)
                u = -1*self.fixed_parameters[location_key]
                bessel = np.abs(scipy.special.j0(k))
                val = np.exp( k*np.cos(x-u) ) / (2*np.pi*bessel)
                vonmises_terms.append(magnitude*val)
            
        vonmises_terms.append(np.ones_like(x)*np.abs(self.fixed_parameters['baseline']))
        val = np.vstack(vonmises_terms).sum(axis=0)
        return val
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = np.diff(edges) + edges[0:-1]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            print len(inputs), len(self.get_val(inputs))
            ax.plot(inputs, data, '.')
            ax.plot(np.array(inputs), self.get_val(inputs), 'r')
            
######################################################################################
# Sum of 3 Von Mises with fixed locations
######################################################################################

class Sumof3x1DVonMisesModelFixedPos(ModelBase):
    def __init__(self, num_vonmises=2, parameters=None, fixed_parameters=None):
        self.num_vonmises = num_vonmises
        self.init_parameters(parameters, fixed_parameters)
        
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {}
            fixed_parameters = {}
            for n in range(self.num_vonmises):
                magnitude_key = 'magnitude_' + str(n)
                location_key = 'location_' + str(n)
                concentration_key = 'concentration_' + str(n)
                fixed_parameters.setdefault(location_key, 0)
                parameters.setdefault(concentration_key, 1)
                parameters.setdefault(magnitude_key, 1)
            fixed_parameters.setdefault('baseline', .5)
        self.parameters = parameters
        self.fixed_parameters = fixed_parameters
        
    def get_val(self, x):
        if type(x) is list:
            x = np.array(x)
        vonmises_terms = []
        for i in range(self.num_vonmises):
        
            magnitude_key = 'magnitude_' + str(i)
            location_key = 'location_' + str(i)
            concentration_key = 'concentration_' + str(i)
            u = self.fixed_parameters[location_key]
            
            try:
                k = np.abs(self.fixed_parameters[concentration_key])
            except:  
                k = np.abs(self.parameters[concentration_key])
            try:
                magnitude = np.abs(self.fixed_parameters[magnitude_key])
            except:
                magnitude = np.abs(self.parameters[magnitude_key])
                
            bessel = np.abs(scipy.special.j0(k))
            val = np.exp( k*np.cos(x-u) ) / (2*np.pi*bessel)
            vonmises_terms.append(magnitude*val)
            
        
        vonmises_terms.append(np.ones_like(x)*np.abs(self.fixed_parameters['baseline']))
        val = np.vstack(vonmises_terms).sum(axis=0)
        return val
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = np.diff(edges) + edges[0:-1]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            print len(inputs), len(self.get_val(inputs))
            ax.plot(inputs, data, '.')
            ax.plot(np.array(inputs), self.get_val(inputs), 'r')

######################################################################################
# Log Normals
######################################################################################

def get_ndim_lognormal_model(n):
    if n == 1:
        return LogNormalModel1D()
    else:
        raise ValueError('not implimented')
                    
class LogNormalModelND(ModelBase):
    '''
    models a lognormal + gaussian distribution
    
    '''
    def __init__(self, dim=1, parameters=None):
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
        terms = []
        for i in range(self.dim):
            mean_key = 'mean_' + str(i)
            std_key = 'std_' + str(i)
            mean = self.parameters[mean_key]
            std = np.abs(self.parameters[std_key])
            
            exponent = ((np.log(inputs[i])-mean)**2)/(2*std**2)
            coeff = 1/(inputs[i]*np.sqrt(2*np.pi*std**2))
            lognorm_term = coeff*np.exp(-1*exponent)
            terms.append(magnitude*lognorm_term)
        # NOTE: This is *NOT* correct for ndim (it works properly for 1 dimensional)
        if self.dim != 1:
            raise ValueError('ndim not implimented!, ignorming dims beyond 1')
        val = terms[0]
        return val
        
class LogNormalModel1D(LogNormalModelND):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = [np.diff(edges) + edges[0:-1]]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(inputs[0], data, '.')
            ax.plot(inputs[0], self.get_val(inputs), 'r')
        
        
######################################################################################
# Gaussian + Log Normals
######################################################################################

def get_ndim_gaussianlognormal_model(n):
    if n == 1:
        return GaussianLogNormalModel1D()
    else:
        raise ValueError('not implimented')
                    
class GaussianLogNormalModelND(ModelBase):
    '''
    models a lognormal + gaussian distribution
    
    '''
    def __init__(self, dim=1, parameters=None):
        self.dim = dim
        self.init_parameters(parameters)
        
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {'lognormmagnitude': 1}
            parameters.setdefault('gaussmagnitude', 1)
            for n in range(self.dim):
                mean_key = 'lognormmean_' + str(n)
                std_key = 'lognormstd_' + str(n)
                stdgauss_key = 'gaussstd_' + str(n)
                meangauss_key = 'gaussmean_' + str(n)
                parameters.setdefault(mean_key, 0)
                parameters.setdefault(std_key, 1)
                parameters.setdefault(stdgauss_key, 0.2)
                parameters.setdefault(meangauss_key, 0)
        self.parameters = parameters
        
    def get_val(self, inputs):
        if type(inputs) is not list:
            inputs = [inputs]
        gaussmagnitude = self.parameters['gaussmagnitude']
        lognormmagnitude = self.parameters['lognormmagnitude']
        terms = []
        for i in range(self.dim):
            mean_key = 'lognormmean_' + str(i)
            std_key = 'lognormstd_' + str(i)
            stdgauss_key = 'gaussstd_' + str(i)
            meangauss_key = 'gaussmean_' + str(i)
            mean = self.parameters[mean_key]
            std = np.abs(self.parameters[std_key])
            stdgauss = np.abs(self.parameters[stdgauss_key])
            meangauss = np.abs(self.parameters[meangauss_key])
            
            exponent = ((np.log(inputs[i])-mean)**2)/(2*std**2)
            coeff = 1/(inputs[i]*np.sqrt(2*np.pi*std**2))
            gauss_term = np.exp(-1*(0.5*(inputs[i] - meangauss)/stdgauss)**2)
            lognorm_term = coeff*np.exp(-1*exponent) + gauss_term
            terms.append(lognormmagnitude*lognorm_term + gaussmagnitude*gauss_term)
        # NOTE: This is *NOT* correct for ndim (it works properly for 1 dimensional)
        if self.dim != 1:
            raise ValueError('ndim not implimented!, ignorming dims beyond 1')
        val = terms[0]
        return val
        
    def get_gaussian_component(self):
        gm = GaussianModel1D()
        for param, val in self.parameters.items():
            if param[0:5] == 'gauss':
                gkey = param[5:]
                gm.parameters[gkey] = val
        return gm
    
    def get_lognormal_component(self):
        lnm = LogNormalModel1D()
        for param, val in self.parameters.items():
            if param[0:7] == 'lognorm':
                lkey = param[7:]
                lnm.parameters[lkey] = val
        return lnm
        
    
class GaussianLogNormalModel1D(GaussianLogNormalModelND):
    def __init__(self, parameters=None):
        self.dim = 1
        self.init_parameters(parameters)
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = [np.diff(edges) + edges[0:-1]]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(inputs[0], data, '.')
            ax.plot(inputs[0], self.get_val(inputs), 'r')
            
    def get_peak(self, min_guess=0.01, max_guess=10, resolution=1000):
        l = np.linspace(min_guess, max_guess, resolution)
        vals = self.get_val(l)
        index = np.argmax(vals)
        return l[index]
        
######################################################################################
# Sum of Gaussians
######################################################################################

class Sumof1DGaussiansModel(ModelBase):
    def __init__(self, num_gaussians=2, parameters=None):
        self.num_gaussians = num_gaussians
        self.init_parameters(parameters)
        
    def init_parameters(self, parameters):
        if parameters is None:
            parameters = {}
            for n in range(self.num_gaussians):
                magnitude_key = {'magnitude_' + str(n)}
                mean_key = 'mean_' + str(n)
                std_key = 'std_' + str(n)
                parameters.setdefault(mean_key, 0)
                parameters.setdefault(std_key, 1)
                parameters.setdefault(magnitude_key, 1)
        self.parameters = parameters
        
    def get_val(self, x):
        if type(inputs) is not list:
            inputs = [inputs]
        gauss_terms = []
        for i in range(self.num_gaussians):
            magnitude_key = {'magnitude_' + str(i)}
            mean_key = 'mean_' + str(i)
            std_key = 'std_' + str(i)
            mean = self.parameters[mean_key]
            std = np.abs(self.parameters[std_key])
            magnitude = self.parameters[magnitude_key]
            gauss_term = (0.5*(x - mean)/std)**2
            gauss_terms.append(magnitude*np.exp(-1*sum(gauss_term)))
        val = np.sum(gaus_terms)
        return val
        
    def fit_occurences(self, x, bins=40, plot=False, method='optimize', density=True):
        data, edges = np.histogram(x, bins=bins, density=density)
        inputs = [np.diff(edges) + edges[0:-1]]
        self.fit(data, inputs, method=method)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(inputs[0], data, '.')
            ax.plot(inputs[0], self.get_val(inputs), 'r')

######################################################################################
# Turbulent mixing
######################################################################################

class TurbulentMixing(ModelBase):
    '''
    Based on Mark Denny's "Biology and the Mechanics of the Wave-Swept Environment", 1988, pg. 147 eqn 10.34
    
    '''
    def __init__(self, dim=3, parameters=None, fixed_parameters=None):
        self.dim = dim
        self.init_parameters(parameters, fixed_parameters)
        
    def init_parameters(self, parameters, fixed_parameters):
        if parameters is None:
            parameters = {'Q': 68}
            parameters.setdefault('alphay', 2)
            parameters.setdefault('alphaz', 2)
            parameters.setdefault('xsource', -.588)
            parameters.setdefault('ustar', .4)
        self.parameters = parameters

        if fixed_parameters is None:
            fixed_parameters = {'u': 0.4}
            #fixed_parameters.setdefault('xsource', -.588)
            fixed_parameters.setdefault('ysource', .003)
            fixed_parameters.setdefault('zsource', .011)
            fixed_parameters.setdefault('background_value', 400)
        self.fixed_parameters = fixed_parameters
        
    def get_val(self, inputs):
        x = inputs[0]
        y = inputs[1]
        z = inputs[2]
        
        Q = np.abs(self.parameters['Q'])
        alphay = np.abs(self.parameters['alphay'])
        alphaz = np.abs(self.parameters['alphaz'])
        u = self.fixed_parameters['u']
        ustar = np.abs(self.parameters['ustar'])
        xsource = self.parameters['xsource']
        ysource = self.fixed_parameters['ysource']
        zsource = self.fixed_parameters['zsource']
        background_value = self.fixed_parameters['background_value']
        
        yterm = (y-ysource)**2*u**2 / (2*alphay**2*ustar**2*(x-xsource)**2)
        zterm = (z-zsource)**2*u**2 / (2*alphaz**2*ustar**2*(x-xsource)**2)

        c = (Q*u) / (2*np.pi*alphay*alphaz*ustar**2*(x-xsource)**2) * np.exp(-1*(yterm + zterm)) + background_value

        return c
        
    def get_array_2d_slice(self, xlim, ylim, zlim, resolution, axis=2, axis_slice=0):
        if axis == 0:
            y = np.arange(ylim[0], ylim[1], resolution, np.float32)
            z = np.arange(zlim[0], zlim[1], resolution, np.float32)[:,np.newaxis]
            x = np.ones_like(y)*axis_slice
            inputs = [x,y,z]
            extent = [np.min(y), np.max(y), np.min(z), np.max(z)]
            return self.get_val(inputs), extent
        
        if axis == 1:
            x = np.arange(xlim[0], xlim[1], resolution, np.float32)
            z = np.arange(zlim[0], zlim[1], resolution, np.float32)[:,np.newaxis]
            y = np.ones_like(x)*axis_slice
            inputs = [x,y,z]
            extent = [np.min(x), np.max(x), np.min(z), np.max(z)]
            return self.get_val(inputs), extent
        
        if axis == 2:
            x = np.arange(xlim[0], xlim[1], resolution, np.float32)
            y = np.arange(ylim[0], ylim[1], resolution, np.float32)[:,np.newaxis]
            z = np.ones_like(x)*axis_slice
            inputs = [x,y,z]
            extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
            return self.get_val(inputs), extent

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
    x = np.random.normal(0, 1, 10000)
    y = np.random.normal(3, 20, 10000)
    z = np.random.normal(0, 2, 10000)
    
    gm = GaussianModel3D()
    gm.fit_occurences([x, y, z], bins=50)
    gm.show_fit(resolution=0.01, axis_slice_threshold=0.1, axis=2, n_inputs_to_show=150)
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
    
def example_sigmoid():
    x = np.linspace(-10,10,100)
    
    real_parameters = {'tao': 10, 'gain': 2, 'xshift':2, 'yshift': 0}
    real_sigmoid = SigmoidModel(real_parameters)
    exact_data = real_sigmoid.get_val(x)
    maxnoise = 0.1*np.max(exact_data)
    noise = np.array([maxnoise*(np.random.random()*2-1) for i in range(len(x))])
    noisy_data = exact_data + noise
    
    fit_sigmoid = SigmoidModel()
    fit_sigmoid.fit(noisy_data, x)
    fit_data = fit_sigmoid.get_val(x)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(x,noisy_data,'.', color='blue')
    ax.plot(x,exact_data,color='blue')
    ax.plot(x,fit_data,color='red')
    
    
