#!/usr/bin/env python

import pytest
import numpy as np
from src.models.perceptron import Perceptron as pn

def test_perceptron_update1():
    x = np.array([0,1])
    y = -1
    w = np.array([1,1])
    model = pn(x, y)
    w1 = model.update(x, y, w)
    return (w1.reshape(-1,) == np.array([1,0])).all()

def test_perceptron_update2(): 
    x = np.random.rand(25)
    y = 1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1-x)<1e-8


def test_perceptron_update3():
    x = np.random.rand(25)
    y = -1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1+x)<1e-8   

def test_Perceptron1():
    N = 100;
    d = 10;
    x = np.random.rand(N,d)
    w = np.random.rand(1,d)
    y = np.sign(w.dot(x.T))[0]
    w, b = perceptron(x,y)
    preds = classify_linear_grader(x,w,b)
    return np.array_equal(preds.reshape(-1,),y.reshape(-1,))


def test_Perceptron2():
    x = np.array([ [-0.70072, -1.15826],  [-2.23769, -1.42917],  [-1.28357, -3.52909],  [-3.27927, -1.47949],  [-1.98508, -0.65195],  [-1.40251, -1.27096],  [-3.35145,-0.50274],  [-1.37491,-3.74950],  [-3.44509,-2.82399],  [-0.99489,-1.90591],   [0.63155,1.83584],   [2.41051,1.13768],  [-0.19401,0.62158],   [2.08617,4.41117],   [2.20720,1.24066],   [0.32384,3.39487],   [1.44111,1.48273],   [0.59591,0.87830],   [2.96363,3.00412],   [1.70080,1.80916]])
    y = np.array([1]*10 + [-1]*10)
    w, b =perceptron(x,y)
    preds = classify_linear_grader(x,w,b)
    return np.array_equal(preds.reshape(-1,),y.reshape(-1,))

def test_linear1():
    xs = np.random.rand(50000,20)-0.5 # draw random data 
    w0 = np.random.rand(20)
    b0 =- 0.1 # with bias -0.1
    ys = classify_linear(xs,w0,b0)
    uniquepredictions=np.unique(ys) # check if predictions are only -1 or 1
    return set(uniquepredictions)==set([-1,1])

def test_linear2():
    xs = np.random.rand(1000,2)-0.5 # draw random data 
    w0 = np.array([0.5,-0.3]) # define a random hyperplane 
    b0 =- 0.1 # with bias -0.1
    ys = np.sign(xs.dot(w0)+b0) # assign labels according to this hyperplane (so you know it is linearly separable)
    return (all(np.sign(ys*classify_linear(xs,w0,b0))==1.0))  # the original hyperplane (w0,b0) should classify all correctly