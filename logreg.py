import numpy as np
from numpy import exp, log, log10, logspace, zeros
from numpy.random import randn, choice
from scipy.linalg import orth
    
def buildmat(m,n,cond_number):
    """Build an mxn matrix with condition number cond."""
    if m<=n:
        U = randn(m,m);
        U = orth(U);
        Vt = randn(n, m);
        Vt = orth(Vt).T;
        S = 1/logspace(0,log10(cond_number),num=m);
        return (U*S[:,None]).dot(Vt)
    else:
        return buildmat(n,m,cond_number).T
    
def create_classification_problem(num_data, num_features, cond_number):
    """Build a simple classification problem."""
    X = buildmat(num_data, num_features, cond_number)
    # The linear dividing line between the classes
    w =  randn(num_features,1)
    # create labels
    prods = X@w
    y = np.sign(prods)
    #  mess up the labels on 10% of data
    flip = choice(range(num_data),int(num_data/10))
    y[flip] = -y[flip]
    #  return result
    return X,y
 
def logistic_loss(z):
    """Return sum(log(1+exp(-z))). Your implementation can NEVER exponentiate a positive number.  No for loops."""
    loss = zeros(z.shape)
    loss[z>=0] = log(1+exp(-z[z>=0]))
    # Make sure we only evaluate exponential on negative numbers
    loss[z<0] = -z[z<0]+log(1+exp(z[z<0]))
    return np.sum(loss)

def logreg_objective(w,X,y):
    """Evaluate the logistic regression loss function on the data and labels, where the rows of D contain 
    feature vectors, and y is a 1D vector of +1/-1 labels."""
    z = y*(X@w)
    return logistic_loss(z)

def logistic_loss_grad(z):
    """Gradient of logistic loss"""
    grad = zeros(z.shape)
    neg = z.ravel() <=0
    pos = z.ravel() > 0
    grad[neg] = -1/(1+exp(z[neg]))
    grad[pos] = -exp(-z[pos])/(1+exp(-z[pos]))
    return grad

def logreg_objective_grad(w,X,y):
    return X.T@(y*logistic_loss_grad(y*X@w))