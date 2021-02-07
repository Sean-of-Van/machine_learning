import numpy as np

def l2distance(X,Z=None):
    '''
    Compute the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    
    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    
    call with only one input:
    l2distance(X)=l2distance(X,X)
    '''
    if Z is None:
        Z=X

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"
    
    G = X @ Z.T   
    S = np.tile(np.diag(X @ X.T), (m,1)).T
    R = np.tile(np.diag(Z @ Z.T), (n,1))
    
    D2 = S + R - 2 * G
    D = D2.copy()
    D[D < 0.0000000001] = 0
    D = np.sqrt(D)
    return D

def findknn(xTr,xTe,k):
    '''
    Find the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    '''

    dists_all = l2distance(xTr, xTe)
    indices = np.argsort(dists_all, axis = 0)[:k]
    dists = np.sort(dists_all, axis = 0)[:k]
    
    return indices, dists

def accuracy(truth,preds):
    '''   
    Analyze the accuracy of a prediction against the ground truth
    
    Input:
    truth = n-dimensional vector of true class labels
    preds = n-dimensional vector of predictions
    
    Output:
    accuracy = scalar (percent of predictions that are correct)
    '''
    
    truth = truth.flatten()
    preds = preds.flatten()

    return (sum(truth == preds)/len(truth))

def mode(datalist):
    '''
    Find the mode of a list of data
    '''
    contents = set(datalist)
    maximum = 0
    for c in contents:
        num = datalist.count(c)
        if num > maximum:
            maximum = num
            m = c
    return m

def modeC(array):
    '''
    Find the mode in each column of an array
    '''
    m = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        m[i] = mode(list(array[:,i]))
    return m

def knnclassifier(xTr,yTr,xTe,k):
    '''   
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    '''

    yTr = yTr.flatten()

    ind, dist = findknn(xTr,xTe,k)

    return (modeC(yTr[ind]))