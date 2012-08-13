# written by Floris van Breugel, with some help from Andrew Straw and Will Dickson
# dependencies for LaTex rendering: texlive, ghostscript, dvipng, texlive-latex-extra

# general imports
import matplotlib
print matplotlib.__version__
print 'recommended version: 1.1.0'
# recommend using version 1.1.0

###################################################################################################
# Floris' parameters for saving figures. 
# NOTE: this could mess up your default matplotlib setup, but it allows for saving to pdf. This currently only works with v1.0.1
# You must import this file before importing matplotlib.pyplot
# see readme
###################################################################################################


###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# used for colorline
from matplotlib.collections import LineCollection

# used in histogram
from scipy.stats import norm as gaussian_distribution
from scipy import signal

# used in colorbar
import matplotlib.colorbar

# used in scatter
from matplotlib.collections import PatchCollection



# not used
#import scipy.optimize
#import scipy.stats.distributions as distributions

###################################################################################################
# Misc Info
###################################################################################################

# FUNCTIONS contained in this file: 
# adjust_spines
# colorline
# histogram
# histogram2d (heatmap)
# boxplot
# colorbar (scale for colormap stuff), intended for just generating a colorbar for use in illustrator figure assembly


# useful links:
# colormaps: http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps




###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

# NOTE: smart_bounds is disabled (commented out) in this function. It only works in matplotlib v >1.
# to fix this issue, try manually setting your tick marks (see example below) 
def adjust_spines(ax,spines, spine_locations={}, smart_bounds=True, xticks=None, yticks=None):
    if type(spines) is not list:
        spines = [spines]
        
    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    if yticks is None:
        yticks = ax.get_yticks()
        
    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]
        
    if 'none' in spines:
        for loc, spine in ax.spines.iteritems():
            spine.set_color('none') # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return
    
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',spine_locations_dict[loc])) # outward by x points
            spine.set_color('black')
        else:
            spine.set_color('none') # don't draw spine
            
    # smart bounds, if possible
    if int(matplotlib.__version__[0]) > 0 and smart_bounds: 
        for loc, spine in ax.spines.items():
            if loc in ['left', 'right']:
                ticks = yticks
            if loc in ['top', 'bottom']:
                ticks = xticks
            spine.set_bounds(ticks[0], ticks[-1])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])    
    
    if 'left' in spines or 'right' in spines:
        ax.set_yticks(yticks)
    if 'top' in spines or 'bottom' in spines:
        ax.set_xticks(xticks)
    
    for line in ax.get_xticklines() + ax.get_yticklines():
        #line.set_markersize(6)
        line.set_markeredgewidth(1)
                
                
        
def adjust_spines_example():
    
    x = np.linspace(0,100,100)
    y = x**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    adjust_spines(ax, ['left', 'bottom'])
    fig.savefig('adjust_spines_example.pdf', format='pdf')
    
    
def adjust_spines_example_with_custom_ticks():

    x = np.linspace(0,100,100)
    y = x**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    
    # set limits
    ax.set_xlim(0,100)
    ax.set_ylim(0,20000)
    
    # set custom ticks and tick labels
    xticks = [0, 10, 25, 50, 71, 100] # custom ticks, should be a list
    adjust_spines(ax, ['left', 'bottom'], xticks=xticks, smart_bounds=True)
    
    ax.set_xlabel('x axis, custom ticks\ncoooool!')
    
    fig.savefig('adjust_spines_custom_ticks_example.pdf', format='pdf')



###################################################################################################
# Colorline
###################################################################################################

# plot a line in x and y with changing colors defined by z, and optionally changing linewidths defined by linewidth
def colorline(ax, x,y,z,linewidth=1, colormap='jet', norm=None, zorder=1, alpha=1, linestyle='solid'):
        cmap = plt.get_cmap(colormap)
        
        if type(linewidth) is list or type(linewidth) is np.array or type(linewidth) is np.ndarray:
            linewidths = linewidth
        else:
            linewidths = np.ones_like(z)*linewidth
        
        if norm is None:
            norm = plt.Normalize(np.min(z), np.max(z))
        else:
            norm = plt.Normalize(norm[0], norm[1])
        
        '''
        if self.hide_colorbar is False:
            if self.cb is None:
                self.cb = matplotlib.colorbar.ColorbarBase(self.ax1, cmap=cmap, norm=norm, orientation='vertical', boundaries=None)
        '''
            
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments, linewidths=linewidths, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha, linestyles=linestyle )
        lc.set_array(z)
        lc.set_linewidth(linewidth)
        
        ax.add_collection(lc)

def colorline_example():
    
    def tent(x):
        """
        A simple tent map
        """
        if x < 0.5:
            return x
        else:
            return -1.0*x + 1
    
    pi = np.pi
    t = np.linspace(0, 1, 200)
    y = np.sin(2*pi*t)
    z = np.array([tent(x) for x in t]) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # standard colorline
    colorline(ax,t,y,z)
    
    # colorline with changing widths, shifted in x
    colorline(ax,t+0.5,y,z,linewidth=z*5)
    
    # colorline with points, shifted in x
    colorline(ax,t+1,y,z, linestyle='dotted')
    
    # set the axis to appropriate limits
    adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlim(0,2)
    ax.set_ylim(0,1.5)
       
    fig.savefig('colorline_example.pdf', format='pdf')
    
    
###################################################################################################
# Histograms
###################################################################################################
    
# first some helper functions
def custom_hist_rectangles(hist, leftedges, width, facecolor='green', edgecolor='none', alpha=1):
    linewidth = 1
    if edgecolor == 'none':
        linewidth = 0 # hack needed to remove edges in matplotlib.version 1.0+

    if type(width) is not list:
        width = [width for i in range(len(hist))]
    rects = [None for i in range(len(hist))]
    for i in range(len(hist)):
        rects[i] = patches.Rectangle( [leftedges[i], 0], width[i], hist[i], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    return rects

def bootstrap_histogram(xdata, bins, normed=False, n=None, return_raw=False):
    if type(xdata) is not np.ndarray:
        xdata = np.array(xdata)

    if n is None:  
        n = len(xdata)
    hist_list = np.zeros([n, len(bins)-1])
    
    for i in range(n):
        # Choose #sample_size members of d at random, with replacement
        choices = np.random.random_integers(0, len(xdata)-1, n)
        xsample = xdata[choices]
        hist = np.histogram(xsample, bins, normed=normed)[0].astype(float)
        hist_list[i,:] = hist
        
    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)
    
    if return_raw:
        return hist_list
    else:
        return hist_mean, hist_std
        
    
def histogram(ax, data_list, bins=10, bin_width_ratio=0.6, colors='green', edgecolor='none', bar_alpha=0.7, curve_fill_alpha=0.4, curve_line_alpha=0.8, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, bootsrap_line_width=0.5, exponential_histogram=False, smoothing_range=None, binweights=None):
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range
    
    if type(bar_alpha) is not list:
        bar_alpha = [bar_alpha for i in range(len(colors))]
    
    n_bars = float(len(data_list))
    if type(bins) is int:
    
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])
        
        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)
        
    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]
        
    bin_centers = np.diff(bins)/2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1-bin_width_ratio)*bin_width/2.
    bar_width = (bin_width-2*bin_width_buff)/n_bars
    
    butter_b, butter_a = signal.butter(curve_butter_filter[0], curve_butter_filter[1])
    
    if return_vals:
        data_hist_list = []
        data_curve_list = []
        data_hist_std_list = []
        
    # first get max number of occurences
    max_occur = []
    for i, data in enumerate(data_list):
        data_hist = np.histogram(data, bins=bins, normed=normed, weights=None)[0].astype(float)
        max_occur.append(np.max(data_hist))
    max_occur = np.max(np.array(max_occur))
        
    for i, data in enumerate(data_list):
        
        if bootstrap_std:
            data_hist, data_hist_std = bootstrap_histogram(data, bins=bins, normed=normed)
        else:
            data_hist = np.histogram(data, bins=bins, normed=normed)[0].astype(float)
            
        if binweights is not None:
            data_hist *= binweights[i]
            if normed:
                data_hist /= np.sum(binweights[i])
            
        if exponential_histogram:
            data_hist = np.log(data_hist+1)
            
        if normed_occurences is not False:
            if normed_occurences == 'total':
                data_hist /= max_occur 
                if bootstrap_std:
                    data_hist_std /= max_occur
            else:
                div = float(np.max(data_hist))
                print div
                data_hist /= div 
                if bootstrap_std:
                    data_hist_std /= div
                    
        
        rects = custom_hist_rectangles(data_hist, bins[0:-1]+bar_width*i+bin_width_buff, width=bar_width, facecolor=colors[i], edgecolor=edgecolor, alpha=bar_alpha[i])
        if bootstrap_std:
            for j, s in enumerate(data_hist_std):
                x = bins[j]+bar_width*i+bin_width_buff + bar_width/2.
                #ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i], linewidth=bootsrap_line_width)
                
                #ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                #ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i])
        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)
        
                
        if show_smoothed:
            if smoothing_range is not None: # in case you only want to smooth and show smoothing over a select range.
                indices_in_smoothing_range = np.where( (bin_centers>smoothing_range[0])*(bin_centers<smoothing_range[-1]) )[0].tolist()
            else:
                indices_in_smoothing_range = [bc for bc in range(len(bin_centers))]
                
            data_hist_filtered = signal.filtfilt(butter_b, butter_a, data_hist[indices_in_smoothing_range])
            interped_bin_centers = np.linspace(bin_centers[indices_in_smoothing_range[0]]-bin_width/2., bin_centers[indices_in_smoothing_range[-1]]+bin_width/2., 100, endpoint=True)
            v = 100 / float(len(bin_centers))
            
            interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range], data_hist_filtered)
            interped_data_hist_filtered2 = signal.filtfilt(butter_b/v, butter_a/v, interped_data_hist_filtered)
            #ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
            if curve_fill_alpha > 0:
                ax.fill_between(interped_bin_centers, interped_data_hist_filtered2, np.zeros_like(interped_data_hist_filtered2), color=colors[i], alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
            if curve_line_alpha:
                ax.plot(interped_bin_centers, interped_data_hist_filtered2, color=colors[i], alpha=curve_line_alpha)
        
        if return_vals:
            data_hist_list.append(data_hist)
            if bootstrap_std:
                data_hist_std_list.append(data_hist_std)
            
            if show_smoothed:
                data_curve_list.append([interped_bin_centers, interped_data_hist_filtered2])
                
    if return_vals and bootstrap_std is False:
        return bins, data_hist_list, data_curve_list
    elif return_vals and bootstrap_std is True:
        return bins, data_hist_list, data_hist_std_list, data_curve_list
    
def histogram_example():
    
    # generate a list of various y data, from three random gaussian distributions
    y_data_list = []
    for i in range(3):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    nbins = 40 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins = np.linspace(-10,30,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    histogram(ax, y_data_list, bins=bins, bin_width_ratio=0.8, colors=['green', 'black', 'orange'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)
    
    adjust_spines(ax, ['left', 'bottom'])
    
    fig.savefig('histogram_example.pdf', format='pdf')
    
###################################################################################################
# Boxplots
###################################################################################################
    
def boxplot(ax, x_data, y_data_list, nbins=50, colormap='YlOrRd', colorlinewidth=2, boxwidth=1, boxlinecolor='black', classic_linecolor='gray', usebins=None, boxlinewidth=0.5, outlier_limit=0.01, norm=None, use_distribution_for_linewidth=False, show_outliers=True, show_whiskers=True):    
    # if colormap is None: show a line instead of the 1D histogram (ie. a normal boxplot)
    # use_distribution_for_linewidth will adjust the linewidth according to the histogram of the distribution
    # classic_linecolor: sets the color of the vertical line that shows the extent of the data, if colormap=None
    # outlier limit: decimal % of top and bottom data that is defined as outliers (0.01 = top 1% and bottom 1% are defined as outliers)
    # show_whiskers: toggle the whiskers, if colormap is None

    if usebins is None: 
        usebins = nbins
        # usebins lets you assign the bins manually, but it's the same range for each x_coordinate

    for i, y_data in enumerate(y_data_list):
        #print len(y_data)

        # calc boxplot statistics
        median = np.median(y_data)
        ind = np.where(y_data<=median)[0].tolist()
        first_quartile = np.median(y_data[ind])
        ind = np.where(y_data>=median)[0].tolist()
        last_quartile = np.median(y_data[ind])
        #print first_quartile, median, last_quartile
        
        # find outliers
        ind_sorted = np.argsort(y_data)
        bottom_limit = int(len(ind_sorted)*(outlier_limit))
        top_limit = int(len(ind_sorted)*(1-outlier_limit))
        indices_inrange = ind_sorted[bottom_limit:top_limit]
        outliers = ind_sorted[0:bottom_limit].tolist() + ind_sorted[top_limit:len(ind_sorted)-1].tolist()
        y_data_inrange = y_data[indices_inrange]
        y_data_outliers = y_data[outliers]
        x = x_data[i]
        
    
        # plot colorline
        if colormap is not None:
            hist, bins = np.histogram(y_data_inrange, usebins, normed=False)
            hist = hist.astype(float)
            hist /= np.max(hist)
            x_arr = np.ones_like(bins)*x
            
            if use_distribution_for_linewidth:
                colorlinewidth = hist*colorlinewidth
                
            colorline(ax, x_arr, bins, hist, colormap=colormap, norm=norm, linewidth=colorlinewidth) # the norm defaults make it so that at each x-coordinate the colormap/linewidth will be scaled to show the full color range. If you want to control the color range for all x-coordinate distributions so that they are the same, set the norm limits when calling boxplot(). 
            
        elif show_whiskers:
            ax.vlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'), linewidth=boxlinewidth/2.)
            ax.vlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'), linewidth=boxlinewidth/2.)
            ax.hlines([np.min(y_data_inrange), np.max(y_data_inrange)], x-boxwidth/4., x+boxwidth/4., color=classic_linecolor, linewidth=boxlinewidth/2.)
            
        
        
        # plot boxplot
        ax.hlines(median, x-boxwidth/2., x+boxwidth/2., color=boxlinecolor, linewidth=boxlinewidth)
        ax.hlines([first_quartile, last_quartile], x-boxwidth/2., x+boxwidth/2., color=boxlinecolor, linewidth=boxlinewidth/2.)
        ax.vlines([x-boxwidth/2., x+boxwidth/2.], first_quartile, last_quartile, color=boxlinecolor, linewidth=boxlinewidth/2.)
        
        # plot outliers
        if show_outliers:
            if outlier_limit > 0:
                x_arr_outliers = x*np.ones_like(y_data_outliers)
                ax.plot(x_arr_outliers, y_data_outliers, '.', markerfacecolor='gray', markeredgecolor='none', markersize=1)
        
    
def boxplot_example():
    # box plot with colorline as histogram

    # generate a list of various y data, from three random gaussian distributions
    x_data = np.linspace(0,20,5)
    y_data_list = []
    for i in range(len(x_data)):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    boxplot(ax, x_data, y_data_list)
    adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    
    fig.savefig('boxplot_example.pdf', format='pdf')    
    
def boxplot_classic_example():
    # classic boxplot look (no colorlines)
        
    # generate a list of various y data, from three random gaussian distributions
    x_data = np.linspace(0,20,5)
    y_data_list = []
    for i in range(len(x_data)):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    boxplot(ax, x_data, y_data_list, colormap=None, boxwidth=1, boxlinewidth=0.5, outlier_limit=0.01, show_outliers=True)
    adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    
    fig.savefig('boxplot_classic_example.pdf', format='pdf')    
    
    
    
###################################################################################################
# 2D "heatmap" Histogram
###################################################################################################

def histogram2d(ax, x, y, bins=100, normed=False, histrange=None, weights=None, logcolorscale=False, colormap='jet', interpolation='nearest', colornorm=None, xextent=None, yextent=None):
    # the following paramters get passed straight to numpy.histogram2d
    # x, y, bins, normed, histrange, weights
    
    # from numpy.histogram2d:
    '''
    Parameters
    ----------
    x : array_like, shape(N,)
      A sequence of values to be histogrammed along the first dimension.
    y : array_like, shape(M,)
      A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array-like or [array, array], optional
      The bin specification:
    
        * the number of bins for the two dimensions (nx=ny=bins),
        * the number of bins in each dimension (nx, ny = bins),
        * the bin edges for the two dimensions (x_edges=y_edges=bins),
        * the bin edges in each dimension (x_edges, y_edges = bins).
    
    range : array_like, shape(2,2), optional
      The leftmost and rightmost edges of the bins along each dimension
      (if not specified explicitly in the `bins` parameters):
      [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
      considered outliers and not tallied in the histogram.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin area.
    weights : array-like, shape(N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i)`. Weights are
      normalized to 1 if normed is True. If normed is False, the values of the
      returned histogram are equal to the sum of the weights belonging to the
      samples falling into each bin.
    '''
    
    hist,x,y = np.histogram2d(x, y, bins, normed=normed, range=histrange, weights=weights)
    
    if logcolorscale:
        hist = np.log(hist+1) # the plus one solves bin=0 issues
        
    if colornorm is not None:
        colornorm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    else:
        colornorm = matplotlib.colors.Normalize(np.min(np.min(hist)), np.max(np.max(hist)))
        
    if xextent is None:
        xextent = [x[0], x[-1]]
    if yextent is None:
        yextent = [y[0], y[-1]]
    
    # make the heatmap
    cmap = plt.get_cmap(colormap)
    ax.imshow(  hist.T, 
                cmap=cmap,
                extent=(xextent[0], xextent[1], yextent[0], yextent[1]), 
                origin='lower', 
                interpolation=interpolation,
                norm=colornorm)
                
def histogram2d_example():  

    # make some random data
    mean = np.random.random()*10
    std = 3
    ndatapoints = 50000
    x = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
    y = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    histogram2d(ax, x, y, bins=100)
    
    adjust_spines(ax, ['left', 'bottom'])
    
    fig.savefig('histogram2d_example.pdf', format='pdf')





###################################################################################################
# Colorbar
###################################################################################################

def colorbar(ax=None, ticks=None, ticklabels=None, colormap='jet', aspect=20, orientation='vertical', filename=None, flipspine=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if ticks is None:
        ticks = np.linspace(-1,1,5,endpoint=True)
    
    ax.set_aspect('equal')
    
    # horizontal
    if orientation == 'horizontal':
        xlim = (ticks[0],ticks[-1])
        yrange = (ticks[-1]-ticks[0])/float(aspect)
        ylim = (0, yrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad,grad))
        if not flipspine:
            adjust_spines(ax,['bottom'], xticks=ticks)
        else:
            adjust_spines(ax,['top'], xticks=ticks)
        if ticklabels is not None:
            ax.set_xticklabels(ticklabels)
    
    # vertical
    if orientation == 'vertical':
        ylim = (ticks[0],ticks[-1])
        xrange = (ticks[-1]-ticks[0])/float(aspect)
        xlim = (0, xrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad,grad)).T
        if not flipspine:
            adjust_spines(ax,['right'], yticks=ticks)
        else:
            adjust_spines(ax,['left'], yticks=ticks)
        if ticklabels is not None:
            ax.set_yticklabels(ticklabels)

    # make image
    cmap = plt.get_cmap(colormap)
    ax.imshow(  im, 
                cmap=cmap,
                extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]), 
                origin='lower', 
                interpolation='bicubic')
                
    if filename is not None:
        fig.savefig(filename, format='pdf')
    
def colorbar_example():
    colorbar(filename='colorbar_example.pdf')
    
###################################################################################################
# Scatter Plot (with PatchCollections of circles) : more control than plotting with 'dotted' style with "plot"
###################################################################################################

def scatter(ax, x, y, color='black', colormap='jet', radius=0.01, colornorm=None, alpha=1, radiusnorm=None, maxradius=1, minradius=0): 
    # color can be array-like, or a matplotlib color 
    # I can't figure out how to control alpha through the individual circle patches.. it seems to get overwritten by the collection. low priority!

    cmap = plt.get_cmap(colormap)
    if colornorm is not None:
        colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)
    
    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    circles = [None for i in range(len(x))]
    for i, pt in enumerate(points):    
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i])*(maxradius-minradius) + minradius
        else:
            r = radius
        circles[i] = patches.Circle( pt, radius=r )

    # make a collection of those circles    
    cc = PatchCollection(circles, cmap=cmap, norm=colornorm) # potentially useful option: match_original=True
    
    # set properties for collection
    cc.set_edgecolors('none')
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)

    # add collection to axis    
    ax.add_collection(cc)  
    
    
def scatter_example():
    
    x = np.random.random(100)
    y = np.random.random(100)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # show a few different scatter examples
    scatter(ax, x, y, color=x*10) # with color scale
    scatter(ax, x+1, y+1, color='black') # set fixed color
    scatter(ax, x+1, y, color='blue', radius=0.05, alpha=0.2) # set some parameters for all circles 
    scatter(ax, x, y+1, color='green', radius=x, alpha=0.6, radiusnorm=(0.2, 0.8), minradius=0.01, maxradius=0.05) # let radius vary with some array 
    
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_aspect('equal')
    adjust_spines(ax, ['left', 'bottom'])
    fig.savefig('scatter_example.pdf', format='pdf')
    
    
###################################################################################################
# Run examples: lets you see all the example plots!
###################################################################################################
def run_examples():
    adjust_spines_example_with_custom_ticks()
    colorline_example()
    histogram_example()
    boxplot_example()
    boxplot_classic_example()
    histogram2d_example()
    colorbar_example()
    scatter_example()

if __name__ == '__main__':
    run_examples()



    
