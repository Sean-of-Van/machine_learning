import numpy as np
from src.models import naive_bayes as nb
from sklearn.naive_bayes import CategoricalNB
import pytest

def hashfeatures(baby, B, FIX):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        B: the number of dimensions to be in the feature vector
        FIX: the number of chunks to extract and hash from each string
    
    Output:
        v: a feature vector representing the input string
    """
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v

def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X

def genTrainFeatures(dimension=128):
    """
    Input: 
        dimension: desired dimension of the features
    Output: 
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """
    
    # Load in the data
    Xgirls = name2features("data/girls.train", B=dimension)
    Xboys = name2features("data/boys.train", B=dimension)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]

def conditional_probXY(X, Y):
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    
    Xpos = np.concatenate([np.ones((1,d)), np.zeros((1,d))])
    Xneg = np.concatenate([np.ones((1,d)), np.zeros((1,d))])
    
    for i, yi in enumerate(Y):
        if yi == 1.0: 
            Xpos = np.concatenate([Xpos, X[i, :][np.newaxis, :]])
        elif yi == -1.0:
            Xneg = np.concatenate([Xneg, X[i, :][np.newaxis, :]])

    pos = np.mean(Xpos, axis=0)
    neg = np.mean(Xneg, axis=0)

    return pos, neg

@pytest.fixture
def get_test_data():
    Xtr, Ytr = genTrainFeatures(128)
    return [(Xtr, Ytr)]

# Check that probabilities sum to 1
def test_naivebayesPY1(get_test_data):
    for data in get_test_data:
        X = data[0]
        Y = data[1]
        model = nb.Naive_Bayes(X, Y)
        pos, neg = model.naivebayesPY()
        assert np.linalg.norm(pos + neg - 1) < 1e-5

# Test the Naive Bayes PY function on a simple example
def test_naivebayesPY2(get_test_data):
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    model = nb.Naive_Bayes(x, y)
    pos, neg = model.naivebayesPY()
    pos0, neg0 = .5, .5
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    assert test < 1e-5

# Test the Naive Bayes PY function on another example
def test_naivebayesPY3(get_test_data):
    x = np.array([[0,1,1,0,1],
        [1,0,0,1,0],
        [1,1,1,1,0],
        [0,1,1,0,1],
        [1,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,0,1]])    
    y = np.array([1,-1, 1, 1,-1,-1, 1])
    model = nb.Naive_Bayes(x, y)
    pos, neg = model.naivebayesPY()
    pos0, neg0 = 5/9., 4/9.
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    assert test < 1e-5

# Tests plus-one smoothing
def test_naivebayesPY4(get_test_data):
    x = np.array([[0,1,1,0,1],[1,0,0,1,0]])    
    y = np.array([1,1])
    model = nb.Naive_Bayes(x, y)
    pos, neg = model.naivebayesPY()
    pos0, neg0 = 3/4., 1/4.
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    assert test < 1e-5    

# test a simple toy example with two points (one positive, one negative)
def test_naivebayesPXY1(get_test_data):
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    model = nb.Naive_Bayes(x, y)
    pos, neg = model.naivebayesPXY()
    pos0, neg0 = conditional_probXY(x,y)
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    assert test < 1e-5

# test the probabilities P(X|Y=+1)
def test_naivebayesPXY2(get_test_data):
    for data in get_test_data:
        X = data[0]
        Y = data[1]    
        model = nb.Naive_Bayes(X, Y)
        pos, neg = model.naivebayesPXY()
        posprobXY, negprobXY = conditional_probXY(X, Y)
        test = np.linalg.norm(pos - posprobXY) 
        assert test < 1e-5

# test the probabilities P(X|Y=-1)
def test_naivebayesPXY3(get_test_data):
    for data in get_test_data:
        X = data[0]
        Y = data[1]    
        model = nb.Naive_Bayes(X, Y)
        pos, neg = model.naivebayesPXY()
        posprobXY, negprobXY = conditional_probXY(X, Y)
        test = np.linalg.norm(neg - negprobXY)
        assert test < 1e-5

# Check that the dimensions of the posterior probabilities are correct
def test_naivebayesPXY4(get_test_data):
    for data in get_test_data:
        X = data[0]
        Y = data[1]
        model = nb.Naive_Bayes(X, Y)
        pos, neg = model.naivebayesPXY()
        posprobXY, negprobXY = conditional_probXY(X, Y)
        assert pos.shape == posprobXY.shape and neg.shape == negprobXY.shape

# test if the log likelihood of the training data are all negative
def test_loglikelihood_neg(get_test_data):
    for data in get_test_data:
        X = data[0]
        Y = data[1]
        model = nb.Naive_Bayes(X, Y)    
        ll = model.loglikelihood(X, Y)
        assert all(ll<0)

# little toy example with two data points (1 positive, 1 negative)
def test_loglikelihood_1():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    model = nb.Naive_Bayes(x, y)    
    ll = model.loglikelihood(x, y)
    assert abs(np.linalg.norm(ll) - 1.14682) < 1e-5

# little toy example with four data points (2 positive, 2 negative)
def test_loglikelihood_2():
    x = np.array([[1,0,1,0,1,1], 
        [0,0,1,0,1,1], 
        [1,0,0,1,1,1], 
        [1,1,0,0,1,1]])
    y = np.array([-1,1,1,-1])
    model = nb.Naive_Bayes(x, y)    
    ll = model.loglikelihood(x, y)
    assert abs(np.linalg.norm(ll) - 5.49449) < 1e-5


# one more toy example with 5 positive and 2 negative points
def test_loglikelihood_3():
    x = np.array([[1,1,1,1,1,1], 
        [0,0,1,0,0,0], 
        [1,1,0,1,1,1], 
        [0,1,0,0,0,1], 
        [0,1,1,0,1,1], 
        [1,0,0,0,0,1], 
        [0,1,1,0,1,1]])
    y = np.array([1, 1, 1 ,1,-1,-1, 1])
    model = nb.Naive_Bayes(x, y)    
    ll = model.loglikelihood(x, y)
    assert abs(np.linalg.norm(ll) - 9.73515) < 1e-5

# X,Y = genTrainFeatures_grader(128)
# posY, negY = naivebayesPY_grader(X, Y)

# check whether the predictions are +1 or neg 1
def test_naivebayes_pred_1(get_test_data):
    for data in get_test_data:
        X = data[0]
        Y = data[1]
        model = nb.Naive_Bayes(X, Y)
        preds = model.naivebayes_pred(X)
        assert np.all(np.logical_or(preds == -1 , preds == 1))

def test_naivebayes_pred_2(get_test_data):
    for data in get_test_data:
        X = data[0][:,:2]
        Y = data[1]
        x_test = np.array([[0,1],[1,0]])
        clf = CategoricalNB()
        clf.fit(X, Y)
        preds_sklrean = clf.predict(x_test)
        model = nb.Naive_Bayes(X, Y)
        preds = model.naivebayes_pred(x_test)

        assert (np.abs(preds - preds_sklrean) < 1e-5).all

def test_naivebayes_pred_3(get_test_data):
    for data in get_test_data:
        X = data[0][:,:6]
        Y = data[1]
        x_test = np.array([[1,0,1,0,1,1], 
            [0,0,1,0,1,1], 
            [1,0,0,1,1,1], 
            [1,1,0,0,1,1]])
        clf = CategoricalNB()
        clf.fit(X, Y)
        preds_sklrean = clf.predict(x_test)
        model = nb.Naive_Bayes(X, Y)
        preds = model.naivebayes_pred(x_test)

        assert (np.abs(preds - preds_sklrean) < 1e-5).all

def test_naivebayes_pred_4(get_test_data):
    for data in get_test_data:
        X = data[0][:,:6]
        Y = data[1]
        x_test = np.array([[1,1,1,1,1,1], 
            [0,0,1,0,0,0], 
            [1,1,0,1,1,1], 
            [0,1,0,0,0,1], 
            [0,1,1,0,1,1], 
            [1,0,0,0,0,1], 
            [0,1,1,0,1,1]])
        clf = CategoricalNB()
        clf.fit(X, Y)
        preds_sklrean = clf.predict(x_test)
        model = nb.Naive_Bayes(X, Y)
        preds = model.naivebayes_pred(x_test)

        assert (np.abs(preds - preds_sklrean) < 1e-5).all