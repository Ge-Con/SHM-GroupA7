from scipy.optimize import minimize
import numpy as np
from prognosticcriteria_v2 import fitness

result_array = []

def ensemble(w0, w1, w2, w3, w4, w5, result_array):
    '''

    Parameters
    ----------
    w0 : TYPE
        weight for 50kHz.
    w1 : TYPE
        weight for 100kHz.
    w2 : TYPE
        weight for 125kHz.
    w3 : TYPE
        weight for 150kHz.
    w4 : TYPE
        weight for 200kHz.
    w5 : TYPE
        weight for 250kHz.
    result_array : TYPE
        2d List containing the HI predictions arrays of each frequency, the first 
        dimension dictates which frequency the result belong to in order from
        lower frequency to high. The second dimension distinguishes between HI
        predictions for All panels index 0, predictions for only the training
        panels, index 1, and finally predictions for only the test panel, index 2.

    Returns
    -------
    avg_test TYPE
        (weighted) Averaged HIs for the test panel
    avg_train
        (weighted) Averaged HIs for the training panels
    avg_full
        All (weighted) averaged HIs

    '''
    exists = False
    for i in range(len(result_array)):
        if not exists:
            all_test = np.expand_dims([result_array[i][2]], axis=2)
            all_train = np.expand_dims([result_array[i][1]], axis=2)
            all_full = np.expand_dims([result_array[i][0]], axis=2)
            exists = True
        else:
            all_test = np.concatenate((all_test, np.expand_dims([result_array[i][2]], axis=2)), axis =2)
            all_train = np.concatenate((all_train,np.expand_dims([result_array[i][1]], axis=2)), axis =2)
            all_full = np.concatenate((all_full, np.expand_dims([result_array[i][0]], axis=2)), axis =2)
    avg_test = np.average(all_test, axis=2, weights= [w0, w1, w2, w3, w4, w5])
    avg_train= np.average(all_train, axis=2, weights= [w0, w1, w2, w3, w4, w5])
    avg_full = np.average(all_full, axis=2, weights= [w0, w1, w2, w3, w4, w5])
    return avg_test[0, :, :], avg_train[0, :, :], avg_full[0, :, :]

def objective_ensemble(*params):
    """
    Objective function for optimizing the ensemble model, uses only the train_HIs to ensure that the model is not fitted on the test data

    Parameters
    ----------
    params : Container of floats
        Model weights: w0, w1, w2, w3, w4, w5.

    Returns
    -------
    error : float
        Error of the fitness function

    """
    print(params)
    ftn, monotonicity, trendability, prognosability, error = fitness(ensemble(*params, result_array)[1])
    return error

def optim2():
    """
    Uses scipy minimize function to find optimal paramters to optimize the weights of the ensemble model.
    TODO: The function it uses and the amount of iterations it performs are to be optimized still

    Returns
    -------
    opt_parameters : List
        list of the optimal parameters .

    """
    res = minimize(objective_ensemble, [0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    opt_parameters = res.x
    print("Best parameters found: ", res.x)
    return opt_parameters

def optimized_ensemble(result_array_dum):
    """
    

    Parameters
    ----------
    result_array_dum : TYPE
        dummy variable to set a global variable,
        2d List containing the HI predictions arrays of each frequency, the first 
        dimension dictates which frequency the result belong to in order from
        lower frequency to high. The second dimension distinguishes between HI
        predictions for All panels index 0, predictions for only the training
        panels, index 1, and finally predictions for only the test panel, index 2.

    Returns
    -------
    TYPE
    avg_test TYPE
        (weighted) Averaged HIs for the test panel
    avg_train
        (weighted) Averaged HIs for the training panels
    avg_full
        All (weighted) averaged HIs

    """
    global result_array
    result_array = result_array_dum
    parameters = optim2()
    return ensemble(*parameters, result_array)







