import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D + 1)
        X = np.concatenate((X, np.ones(len(X)).reshape(-1, 1)), axis=1)
        y = 2 * y - 1
        for i in range(max_iterations):
            sign = y * np.dot(X, w)
            sign = np.where(sign <= 0, 1, 0)
            sign = y.reshape(-1, 1) * sign.reshape(-1, 1)
            w += (step_size / N) * (np.dot(sign.T, X)).flatten()
        b = w[-1]
        w = w[0:D].flatten()
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        y = np.where(y == 1, y, y - 1)
        coe = step_size/N
        for i in range(max_iterations):
            z = y * (np.dot(X, w) + b * np.ones(N))
            sig = sigmoid(-z)
            temp = sig * y
            w += coe * np.dot(temp, X)
            b += coe * np.dot(temp, np.ones(N))
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-1 * z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        result = np.dot(X, w) + b * np.ones(N)
        preds = np.where(result > 0, 1, 0)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        z = np.dot(X, w) + b * np.ones(N)
        prob = sigmoid(z)
        preds = np.where(prob > 0.5, 1, 0)
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        for i in range(max_iterations):
            random_xn = np.random.choice(N)
            xn = X[random_xn]
            yn = y[random_xn]
            gw_matrix = np.dot(w, xn.T)+ b
            exp_gw = np.exp(gw_matrix - np.max(gw_matrix))
            softmax = exp_gw/np.sum(exp_gw)
            softmax[yn] -= 1
            w -= step_size * np.dot(np.mat(softmax).T, np.mat(xn))
            b -= step_size * softmax
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        y = np.eye(C)[y]
        coe = step_size/N
        for i in range(max_iterations):
            gw_matrix = np.dot(w, X.T).T + b
            exp_gw = np.exp(gw_matrix - np.max(gw_matrix))
            softmax = exp_gw.T / np.sum(exp_gw, axis=1)
            softmax = softmax.T - y
            w -= coe * np.dot(softmax.T, X)
            b -= coe * np.sum(softmax, axis=0)

        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    X = np.insert(X, D, values=np.ones(N), axis=1)
    w = np.insert(w, D, values=b, axis=1)
    p = np.matmul(w, X.T)
    preds = np.argmax(p, axis=0)*1.0

    ############################################

    assert preds.shape == (N,)
    return preds




        