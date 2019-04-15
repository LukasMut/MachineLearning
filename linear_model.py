import numpy as np
import random 

from feature_scaling import Scaler
from loss_functions import * 

class MultivarLinReg():
    
    # create initializer method
    def __init__(self, features: str, method: str, loss: str, weights: str, alpha: float, max_iter: int):
        """
        The features property determines whether weights should be computed for one single feature or all features.
        This argument can only be set to 'single' or 'all'.
        
        The method property determines whether gradient descent or an analytic solution should be used for weights
        optimization. This argument can only be set to 'analytic' or 'gradient_descent'.
        
        The loss property determines whether the mean squared error or the root mean squared error should be used as
        the loss function. This argument can only be set to 'mse' or 'rmse'.
        
        The weights property determines whether weights should either be initialized randomly or with zeros.
        This argument can only be set to 'random' or 'zeros'.
        
        The alpha property denotes the learning rate of gradient descent. It controls the step sizes taken
        and thus is responsible for the convergence. Reasonable alpha values are 0.1, 0.01 or 0.001. Try different values.
        
        The max_iter property denotes the number of iterations gradient descent should maximally take until convergence.
        """
        self.features = features
        self.method = method
        self.loss = loss
        self.weights = weights
        self.alpha = alpha
        self.max_iter = max_iter
    
    def preprocessing(self, X):
        """
        Usually a constant is included as one of the regressors due to the offset parameter w_0.
        As w_0 is just an off-set parameter and denotes the intercept, we have to add a column with ones (x_0 = 1)
        to the features matrix X. 

        Gradient descent will take longer to reach the global minimum when the features are not on a similar scale.
        Thus, feature scaling is an important step prior to gradient descent computation.
        
        The data matrix is required to be (centered and) normalized before (!) we add a column with ones (x_0 = 1) to X.
        """
        N = X.shape[0]
        scaler = Scaler(method='normalization')
        # we only use the train set
        X = scaler.normalization(X)

        if self.features == 'single':
            X = np.c_[np.ones(N, dtype = int), X[:,0]]
            return X

        elif self.features == 'all':
            X = np.c_[np.ones(N, dtype = int), X]
            return X

        else:
            raise Exception('The features argument can only be set to "single" or "all"')
        
    def analytic(self, X, y):
        """
        This function computes the optimal weights with an analytic solution instead of using gradient descent.
        """
        if self.method == 'analytic':
            
            X = self.preprocessing(X)
            try:
                inverse_mat = np.linalg.inv(X.T.dot(X))
            except np.linalg.LinAlgError:
                print('The inverse cannot be computed. Look at your data!')

            weights = np.dot(inverse_mat.dot(X.T), y)
            return weights
        
        else:
            raise Exception('The argument "method" is not set to analytic')
                      
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
    
    def loss_computation(self, X, y, w):
        """
        Compute either mean squared error or root mean squared error as in-sample loss per iteration.
        """
        if self.loss == 'mse':
            J = MSE(X, y, w)
            return J

        elif self.loss == 'rmse':
            J = RMSE(X, y, w)
            return J

        else:
            raise Exception('The loss argument can only be set to "mse" or "rmse"')
            
    @staticmethod
    def gradients(X, y, w):
        """
        This function computes the gradients (i.e., derivative with respect to each parameter w_j)
        to update the weights after each iteration.
        It computes the derivative with respect to each w_j simulteanously. 
        
        This function is a static method as it does not access any parameters of the object.
        You just have to pass the features matrix X, the vector y and the weights w. No instance is required.
        """
        n = X.shape[0]
        f = X @ w

        # compute the derivative with respect to theta_j for all theta_j in the weights vector (simultaneously)
        derivatives = [(2 / n) * np.sum((f - y) * X[:,i]) for i, _ in enumerate(w)]

        return derivatives
    
    # default tolerance is set to 0.001 (can easily be changed)
    def gradient_descent(self, X, y, tolerance = 0.001):
        """
        Compute gradient descent to find the optimal weights / coefficients.
        Be aware that w_0 is just an off-set parameter which denotes the intercept and is not a coefficient for a feature.

        Gradient descent will take longer to reach the global minimum when the features are not on a similar scale.
        Thus, feature scaling is an important step prior to gradient descent computation.
        """
                      
        if self.method == 'gradient_descent':
        
            X = self.preprocessing(X)
            weights = self.init_weights(X)
            
            # setting parameters to determine convergence
            alpha = self.alpha
            max_iter = self.max_iter
            num_iter = 1
            convergence = 0
            
            # initialize loss with 0
            loss = 0
            losses = []

            while convergence < 1:

                current_loss = self.loss_computation(X, y, weights)
                losses.append(current_loss) 
                derivatives = MultivarLinReg.gradients(X, y, weights)

                weights = np.array([w - alpha * d for w, d in zip(weights, derivatives)])

                num_iter += 1        
                diff = abs(loss - current_loss)
                loss = current_loss     

                if diff < tolerance:
                    convergence = 1
                elif num_iter > max_iter:
                    convergence = 1 

            return weights, losses
        
        else:
            raise Exception('The method argument is not set to "gradient_descent"')