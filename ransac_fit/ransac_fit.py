import ransac
import matplotlib.pyplot as plt
import numpy as np

def linear_ransac_fit(x, y, min_data_vals=50, max_iterations=10, threshold=2e3, num_vals_req=100, plot=False):
    assert(len(x)==len(y))
    n_samples = len(x)
    all_data = np.vstack((x, np.ones_like(x), y)).T
    
    n_inputs = 2
    n_outputs = 1
    
    input_columns = range(n_inputs) # the first columns of the array
    output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
    linear_model = ransac.LinearLeastSquaresModel(input_columns, output_columns)
    
    # run RANSAC algorithm
    fit, ransac_data = ransac.ransac(all_data, linear_model,
                                     min_data_vals, max_iterations, threshold, num_vals_req, # misc. parameters
                                     debug=False,return_all=True)
                                     
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y,'.')
        
        x_vals = np.linspace(np.min(x), np.max(x), 10)
        data = np.vstack((x_vals, np.ones_like(x_vals), y)).T
        y_vals = linear_model.get_value(data, fit)
        ax.plot(x_vals, y_vals, 'red')
        
        plt.show()
    
    return fit, ransac_data
