import numpy as np

class Scaler():

    # create initializer method
    def __init__(self, method: str):
        """
        The property method determines whether the Scaler should only perform mean centering (i.e., mean = 0)
        or also normalization / standardization (i.e., unit variance).
        Thus this argument can only be set to "centering" or "normalization".
        """
        self.method = method

     
    def scaling(self, X, normalized_X, means: list, stds: list):
        """
        This function computes feature scaling and either centers or normalizes the data (dependent on the setup).
        Usually, data normalization is a crucial step prior to further computations. That's why centering is set to False by default.
        However, for multidimensional scaling / principal component analysis, the data should just be centered.
        """
        for i, feature in enumerate(X):
            for j, x in enumerate(feature):
                if self.method == 'centering':
                    normalized_X[i, j] += (x - means[i])
                elif self.method == 'normalization':
                    normalized_X[i, j] += (x - means[i]) / stds[i]
                else:
                    raise Exception('The method argument can only be set to "centering" or "normalization"')
        
        return normalized_X.T

    def normalization(self, Xtrain, Xtest = list()):
        """
        Normalization is required to be computed using the mean and the std of each feature from the train data. 
        Otherwise you'd overfit.

        If you pass a train and a test set, the function will return a correctly normalized version of both datasets,
        else it will only return a normalized version of the train set. 
        """

        Xtrain = Xtrain.T
        normalized_train = np.zeros_like(Xtrain)

        means, stds = [np.mean(feature) for feature in Xtrain], [np.std(feature) for feature in Xtrain]

        normalized_train = self.scaling(Xtrain, normalized_train, means, stds)

        if len(Xtest) > 0:

            Xtest = Xtest.T
            normalized_test = np.zeros_like(Xtest)
            normalized_test = self.scaling(Xtest, normalized_test, means, stds)
            
            return normalized_train, normalized_test

        else:
            return normalized_train
