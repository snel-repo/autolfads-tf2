import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

def simple_ridge_wrapper(X_3d, y_3d):
    """wrapper that performs affine transformation.
    
    Parameters
    ----------
    X_3d : Predictor
        shape n_trials x n_samples x n_features.
    y_3d : target variables
        shape n_trials x n_samples x n_targets.
    
    Returns
    -------
    r2 : the R2 between true and predicted y_3d for only validation trials

    """
    # set up train and valid inds
    fold = 5
    use_sklearn = True
    n_trials = X_3d.shape[0]
    valid_idx = np.zeros(n_trials, dtype=bool)
    valid_idx[::fold] = True
    train_idx = ~valid_idx
    # split trials into train and validation trials
    train_X_3d = X_3d[train_idx,:,:]
    valid_X_3d = X_3d[valid_idx,:,:]
    train_y_3d = y_3d[train_idx,:,:]
    valid_y_3d = y_3d[valid_idx,:,:]
    # flatten data
    X_dim = X_3d.shape[2]
    y_dim = y_3d.shape[2]
    train_X = train_X_3d.transpose((2, 0, 1)).reshape((X_dim, -1)).T
    valid_X = valid_X_3d.transpose((2, 0, 1)).reshape((X_dim, -1)).T
    train_y = train_y_3d.transpose((2, 0, 1)).reshape((y_dim, -1)).T
    valid_y = valid_y_3d.transpose((2, 0, 1)).reshape((y_dim, -1)).T
    if use_sklearn:
        clf = Ridge(alpha=1e-6)
        clf.fit(train_X, train_y)
        valid_yhat = clf.predict(valid_X)
    else:
        # add bias to the predictors
        train_X = np.c_[ np.ones((np.size(train_X, 0), 1)), train_X ]
        valid_X = np.c_[ np.ones((np.size(valid_X, 0), 1)), valid_X ]
        # perform affine transformation
        w = train_ridge_regression(train_X, train_y)
        train_yhat = np.dot( train_X, w)
        valid_yhat = np.dot( valid_X, w)
    # compute R2. This can be modified to use multiouput='uniform_average' if y is higher-D
    r2 = r2_score(valid_y, valid_yhat, multioutput='raw_values')
    return r2

def train_ridge_regression(X, y, l2=0):
    """Perform ridge regression, if we want to train without sklearn
    
    Parameters
    ----------
    X : Predictor
        shape n_samples x n_features.
    y : target variables
        shape n_samples x n_targets.
    l2 : lambda penalty
        scalar.
    
    Returns
    -------
    W : weights that maps X to y

    """
    R = l2 * np.eye( X.shape[1] )
    R[0,0] = 0
    temp = np.linalg.inv(np.dot(X.T, X) + R)
    temp2 = np.dot(temp,X.T)
    W = np.dot(temp2,y)
    return W

def compute_zig_mean(zig_params, s_min, fs):
    
    ''' INPUTS
        zig_params     -- Previously collapsed data. Size: [3*n_neurons, n_timesteps, n_trials]
        s_min          -- minimum calcium event size. Scalar
        fs             -- Sampling rate in Hz. Scalar
    
        OUTPUT
        rates          -- Estimated event rates. Size: [n_neurons, n_timesteps, n_trials]
    '''
    n_chs = zig_params.shape()[0]/3
    shape = zig_params[0:n_chs-1,:,:]
    scale = zig_params[n_chs:2*n_chs-1,:,:]
    q = zig_params[2*n_chs:-1,:,:]
    rates = fs*(np.multiply(q,(np.multiply(shape,scale)+s_min))) 
    
    return rates, shape, scale, q