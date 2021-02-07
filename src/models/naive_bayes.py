import numpy as np

class Naive_Bayes:
    '''
    Naive Bayes algorithm. Current implementation is Boolean Categories and Boolean Classifications only.

    Data in format:
        X : n independant variables with d dimensions (n x d) of features (0 or 1)
        Y : n dependent variables (n x 1)  of categories (+1 or -1)

    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.pos, self.neg = self.naivebayesPY()
        self.posprob, self.negprob = self.naivebayesPXY()
    
    def naivebayesPY(self):
        '''
        naivebayesPY(Y) returns [pos,neg], the probability of Y = +1 or Y = -1

        Computation of P(Y)
        Input:
            Y : n labels (-1 or +1) (nx1)

        Output:
            pos: probability p(y=1)
            neg: probability p(y=-1)
        '''
        Y = self.Y
        # add one positive and negative example to avoid division by zero ("plus-one smoothing")
        Y = np.concatenate([Y, [-1,1]])
        n = len(Y)
        
        pos = np.sum(Y==1)/n
        neg = np.sum(Y==-1)/n

        return pos, neg

    def naivebayesPXY(self):
        '''
        naivebayesPXY(X, Y) returns [posprob,negprob], the probability that X = 1, for Y = 1 and Y = -1
        
        Input:
            X : n input vectors of d dimensions (nxd)
            Y : n labels (-1 or +1) (n)
        
        Output:
            posprob: probability vector of p(x_alpha = 1|y=1)  (d)
            negprob: probability vector of p(x_alpha = 1|y=-1) (d)
        '''
        X = self.X
        Y = self.Y
        # add one positive and negative example to avoid division by zero ("plus-one smoothing")
        n, d = X.shape
        X = np.concatenate([X, np.ones((2,d)), np.zeros((2,d))])
        Y = np.concatenate([Y, [-1,1,-1,1]])
        
        posprob = np.mean(X[Y==1], axis = 0)
        negprob = np.mean(X[Y==-1], axis = 0)

        return posprob, negprob

    def loglikelihood(self, X_test, Y_test):
        '''
        loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test
        i.e. the log probability that X = 1, for Y = 1 and Y = -1
        
        Input:
            posprob: conditional probabilities for the positive class (d)
            negprob: conditional probabilities for the negative class (d)
            X_test : features (nxd)
            Y_test : labels (-1 or +1) (n)
        
        Output:
            loglikelihood of each point in X_test (n)
        '''
        posprob = self.posprob
        negprob = self.negprob
        
        n, d = X_test.shape
        loglikelihood = np.zeros(n)
        
        probs = np.zeros((n, d))
        
        probs[Y_test == 1,:] = posprob
        probs[Y_test == -1,:] = negprob
        
        probs_elements = np.zeros((n, d))
        
        probs_elements = np.multiply(probs, X_test == 1) + np.multiply(1 - probs, X_test == 0)
        
        loglikelihood = np.sum(np.log(probs_elements), axis = 1)
            
        return loglikelihood

    def naivebayes_pred(self, X_test):
        '''
        naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test
        
        Input:
            pos: class probability for the positive class
            neg: class probability for the negative class
            posprob: conditional probabilities for the positive class (d)
            negprob: conditional probabilities for the negative class (d)
            X_test : features (nxd)
        
        Output:
            prediction of each point in X_test (n)
        '''
        n, d = X_test.shape

        Y_pos = np.ones(n)
        Y_neg = Y_pos * -1
        
        expected = self.loglikelihood(X_test, Y_pos) - self.loglikelihood(X_test, Y_neg)
        
        pred = expected.copy()
        
        pred[expected < 0] = -1
        pred[expected >= 0] = 1
        
        return pred

    def accuracy(self, Ytruth, Ytest):
        '''   
        Analyze the accuracy of a prediction against the ground truth
        
        Input:
        truth = n-dimensional vector of true class labels
        preds = n-dimensional vector of predictions
        
        Output:
        accuracy = scalar (percent of predictions that are correct)
        '''
        
        truth = Ytruth.flatten()
        preds = Ytest.flatten()

        return (sum(truth == preds)/len(truth))