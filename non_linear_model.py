import numpy as np
import random

from feature_scaling import Scaler
from loss_functions import log_error

class LogisticRegression():

    # create initializer method
    def __init__(self, loss: str, weights: str, alpha: float, max_iter: int):
        """
        The loss property determines the error, that will be used as the loss function.
        This argument can only be set to 'log_likelihood' as we want to evaluate the log likelihood error of our model. 
        
        The weights property determines whether weights should either be initialized randomly or with zeros.
        This argument can only be set to 'random' or 'zeros'.
        
        The alpha property denotes the learning rate of gradient descent. It controls the step sizes taken
        and thus is responsible for the convergence. Reasonable alpha values are 0.1, 0.01 or 0.001. Try different values.
        
        The max_iter property denotes the number of iterations gradient descent should maximally take until convergence.
        """

        self.loss = loss

        if self.loss != "log_likelihood":
            raise Exception("Logistic Regression tries to minimize the log likelihood. No other loss can be applied.")

        self.weights = weights
        self.alpha = alpha
        self.max_iter = max_iter

    def preprocessing(self, X):
        """
        Usually a constant is included as one of the regressors due to the offset parameter w_0.
        As w_0 is just an off-set parameter and denotes the intercept, we have to add a column with ones (x_0 = 1)
        to the features matrix X. 

        Gradient descent will take longer to reach the global minimum when the features are not on a similar scale.
        Thus, feature scaling (e.g., mean centering, unit variance) is an important step prior to gradient descent computation (!).
        """
        N = X.shape[0]
        scaler = Scaler(method='normalization')
        # we only use the train set to normalize the data
        X = scaler.normalization(X)
        X = np.c_[np.ones(N, dtype = int), X]
        return X

    def init_weights(self, X):
        """
        Weights are either initialized randomly or with zeros (dependent on the set-up).
        """
        M = X.shape[1]
        
        if self.weights == 'random':
            random.seed(42)
            weights = np.array([random.random() for _ in range(M)])
            return weights
        
        elif self.weights == 'zeros':
            weights = np.zeros(M, dtype = int)
            return weights

        else:
            raise Exception('The weights argument can only be set to "random" or "zeros"')
    
    @staticmethod
    def gradients(X, y, w):
        """
        Compute gradients after each weight update.
        """
        N = X.shape[0]
        g = np.zeros_like(w)

        for n in range(N):
            g -= y[n]*np.exp(-y[n]*(w @ X[n])) * X[n]
        return g / N  


    def gradient_descent(self, Xtrain, ytrain, tolerance = 1e-7):
        """
        Compute gradient descent to find the optimal weights / coefficients.
        Be aware that w_0 is just an off-set parameter which denotes the intercept / bias and is not a coefficient for a feature.

        Gradient descent will take longer to reach the global minimum when the features are not on a similar scale.
        Thus, feature scaling is an important step prior to gradient descent computation (!).
        """

        X = self.preprocessing(Xtrain)

        # y is a N by 1 matrix of target values -1 and 1
        y = np.array((ytrain -.5) * 2)

        # initialize weights
        w = self.init_weights(X)

        # compute log likelihood function
        if self.loss == 'log_likelihood':
            loss = log_error(X, y, w) 

        # initialize learning rate for gradient descent
        learning_rate = self.alpha
        max_iter = self.max_iter

        num_iter = 0  
        convergence = 0

        # keep track of in-sample log losses
        losses = []

        while convergence < 1:
            num_iter += 1                        

            # compute gradient at current w      
            g = LogisticRegression.gradients(X, y, w)
                  
            # take a step into steepest descent (multiply the gradient by the learning rate and subtract that value from w)
            
            w_update = w - learning_rate * g
                       
            # compute in-sample error for new w
            cur_loss = log_error(X, y, w_update)
           
            if cur_loss < loss:
                w = w_update
                loss = cur_loss
                losses.append(loss)
                        
            g_norm = np.linalg.norm(g)

            # examine whether gradient norm is below threshold
            if g_norm < tolerance:
                convergence = 1

            # check whether we have reached max number of iterations
            elif num_iter > max_iter:
                convergence = 1

        return w, losses

    @staticmethod
    def logistic(input):
        """
        Logistic function. Solves a binary classification task.
        """
        pred = 1 / (1 + np.exp(-input))
        return pred

    def predict(self, w, Xtrain, Xtest = list()):
        """
        Prediction function, which takes train and test set. 
        You are required to additionally pass the train set as the test set needs to be normalized based on the train set.
        
        If the logistic function on w^T X[n] is >= 0.5, label 1 else label 0.
        """

        # initialize the scaler
        scaler = Scaler(method='normalization')

        if len(Xtest) == 0:
            N = Xtrain.shape[0]
            Xtrain_norm = scaler.normalization(Xtrain)
            X = np.c_[np.ones(N, dtype = int), Xtrain_norm]

        else:
            N = Xtest.shape[0]
            # fit the scaler on the train data and normalize test data based on the train data
            Xtrain_norm, Xtest_norm = scaler.normalization(Xtrain, Xtest)
            X = np.c_[np.ones(N, dtype = int), Xtest_norm]

        P = np.zeros(N)
        # as we want to predict classes, data should be of type integer
        pred_classes = np.zeros(N, dtype = int)
        
        for n in range(N):
            P[n] = LogisticRegression.logistic(w @ X[n])
            pred_classes[n] = 0 if P[n] < 0.5 else 1 
            
        return pred_classes