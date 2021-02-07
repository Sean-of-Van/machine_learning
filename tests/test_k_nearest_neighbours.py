#!/usr/bin/env python

'''Tests for 'predict_model' package.'''

import pytest

import numpy as np
from src.models import k_nearest_neighbours as knn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

def test_knn_0():
    # checking output types
    xTr = np.random.rand(500,10) # defininng 500 training data points 
    xTe = np.random.rand(300,10) # defining 300 testing data points
    Ig,Dg = knn.findknn(xTr,xTe,5) # compute indices and distances to the 5- nearest neighbors 
    # check if Ig is a matrix of integers, Dg a matrix of floats
    test=(type(Ig)==np.ndarray)  & (type(Ig)==np.ndarray) & ((type(Dg[0][0])==np.float64) or (type(Dg[0][0])==np.float32)) & ((type(Dg[0][0])==np.float64) or (type(Dg[0][0])==np.float32))
    assert test

def test_knn_1():
    # checking output dimensions
    xTr = np.random.rand(500,10) # defininng 500 training data points 
    xTe = np.random.rand(300,10) # defining 300 testing data points
    Ig,Dg = knn.findknn(xTr,xTe,5) # compute indices and distances to the 5- nearest neighbors 
    test=(Ig.shape==(5,300)) & (Dg.shape==(5,300)) # test if output dimensions are correct
    assert test

def test_knn_2():
    # checking 1-NN accuracy
    xTr = np.random.rand(500,10) # defininng 500 training data points 
    xTe = np.random.rand(300,10) # defining 300 testing data points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xTr)
    Dg, Ig = nbrs.kneighbors(xTe)
    Is, Ds = knn.findknn(xTr,xTe,1) 
    test = np.linalg.norm(Ig[:,0] - Is[0,:]) + np.linalg.norm(Dg[:,0] - Ds[0,:]) # compare results
    assert test<1e-5 

def test_knn_3():
    # checking 3-NN accuracy
    xTr = np.random.rand(500,10) # defininng 500 training data points 
    xTe = np.random.rand(300,10) # defining 300 testing data points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xTr)
    Dg, Ig = nbrs.kneighbors(xTe)
    Is,Ds = knn.findknn(xTr,xTe,3)
    test = np.linalg.norm(Ig[:,0] - Is[0,:]) + np.linalg.norm(Dg[:,0] - Ds[0,:]) # compare results
    assert test<1e-5 

def test_accuracy_0():
    # check type of output is correct
    truth = np.array([1, 2, 3, 4])
    preds = np.array([1, 2, 3, 0])
    assert type(knn.accuracy(truth,preds))==np.float64

def test_accuracy_1():
    # accuracy check on 4 sample data
    truth = np.array([1, 2, 3, 4]) # define truth 
    preds = np.array([1, 2, 3, 0]) # define preds
    assert abs(knn.accuracy(truth,preds) - 0.75)<1e-10 # check if accuracy is correct

def test_accuracy_2():
    # accuracy check on random samples
    p=np.random.rand(1,1000) # define random string of [0,1] as truth
    truth=np.int16(p>0.5)
    p2=p+np.random.randn(1,1000)*0.1 # define very similar version as preds
    preds=np.int16(p2>0.5)
    print(accuracy_score(truth, preds))
    print(knn.accuracy(truth,preds))
    assert abs(knn.accuracy(truth,preds) - accuracy_score(truth[0,:],preds[0,:]))<1e-3 # check if accuracy is correct

def test_knn_classifier_0():
    # test if output is a numpy array, and of the right length
    X = np.array([[1,0,0,1],[0,1,0,1]]).T
    y = np.array([1,1,2,2])
    preds = knn.knnclassifier(X,y,X,1)
    assert type(preds)==np.ndarray and preds.shape==(4,)

def test_knn_classifier_1():
    X = np.array([[1,0,0,1],[0,1,0,1]]).T
    y = np.array([1,1,2,2])
    np.testing.assert_allclose(knn.knnclassifier(X,y,X,1),y)
    assert np.testing.assert_allclose

def test_knn_classifier_2():
    X = np.array([[1,0,0,1],[0,1,0,1]]).T
    y = np.array([1,1,2,2])
    y2 = np.array([2,2,1,1])
    assert np.array_equal(knn.knnclassifier(X,y,X,3),y2)

def test_knn_classifier_3():
    X = np.array([[-4,-3,-2,2,3,4]]).T
    y = np.array([1,1,1,2,2,2])
    X2 = np.array([[-1,1]]).T
    y2 = np.array([1,2])
    assert np.array_equal(knn.knnclassifier(X,y,X2,2),y2)

def test_knn_classifier_4():
    X = np.array([[-4,-3,-2,2,3,4]]).T
    y = np.array([1,1,1,2,2,2])
    X2 = np.array([[0,1]]).T
    y2 = np.array([1,2])
    y3 = np.array([2,2])
    assert np.array_equal(knn.knnclassifier(X,y,X2,2),y2) or np.array_equal(knn.knnclassifier(X,y,X2,2),y3)

def test_knn_classifier_5():
    X = np.random.rand(4,4)
    y = np.array([1,2,2,2])
    assert knn.accuracy(knn.knnclassifier(X,y,X,1),y) == 1

def test_knn_classifier_6():
    X = np.random.rand(4,4)
    y = np.array([1,2,1,2])
    assert knn.accuracy(knn.knnclassifier(X,y,X,1),y) == 1

def test_knn_classifier_7():
    X = np.random.rand(10,100)
    y = np.round(np.random.rand(10)).astype('int')
    assert knn.accuracy(knn.knnclassifier(X,y,X,1),y) == 1