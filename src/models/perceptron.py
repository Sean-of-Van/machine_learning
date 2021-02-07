import numpy as np

class Perceptron:

    def __init__(self, xTr, yTr, max_iter=100):
        self.w, self.b = perceptron(xTr, yTr, max_iter)


    def perceptron_update(self, x, y, w):
        '''
        function w=perceptron_update(x,y,w);
        
        Implementation of Perceptron weights updating
        Input:
        x : input vector of d dimensions (d)
        y : corresponding label (-1 or +1)
        w : weight vector of d dimensions
        
        Output:
        w : weight vector after updating (d)
        '''
        
        return w + y * x

    def perceptron(self, xTr, yTr, max_iter=100):
        '''
        function w=perceptron(xs,ys);
        
        Implementation of a Perceptron classifier
        Input:
        xTr : n input vectors of d dimensions (nxd)
        yTr : n labels (-1 or +1)
        
        Output:
        w : weight vector (1xd)
        b : bias term
        '''

        n, d = xTr.shape  
        w = np.zeros(d)
        b = 0.0
        itn = 0
        while itn < max_iter:
            m = 0
            seed = np.random.randint(0, 2**(32 - 1) - 1)
            rstate = np.random.RandomState(seed)
            rstate.shuffle(xTr)
            rstate.shuffle(yTr)
            
            for i in range(n):
                if (yTr[i] * np.dot(w, xTr[i])) <= 0:
                    w = self.perceptron_update(xTr[i],yTr[i],w)
                    m = m + 1
            
            if m == 0:
                break
            
            itn += 1

        return (w,b)

    def classify_linear(self, xs,w,b=None):
        '''
        function preds=classify_linear(xs,w,b)
        
        Make predictions with a linear classifier
        Input:
        xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
        w : weight vector of dimensionality d
        b : bias (scalar)
        
        Output:
        preds: predictions (1xn)
        '''   
        w = w.flatten()    
        predictions=np.zeros(xs.shape[0])
        
        for i, x in enumerate(xs):
            if np.dot(w, xs[i]) + b < 0:
                predictions[i] = -1
            else:
                predictions[i] = 1
                
        return predictions