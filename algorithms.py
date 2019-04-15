import numpy as np
import matplotlib.pyplot as plt

from feature_scaling import Scaler

class GradientDescent():

    # create initializer method
    def __init__(self, alpha: float, max_iter: int, tolerance):
        """
        The property alpha denotes the learning rate. The learning rate determines the step size of gradient descent.
        Reasonable alpha values are 0.1, 0.01 or 0.001. The lower the alpha value is, the smaller steps gradient descent will take.

        The property max_iter denotes the maximum number of iterations (i.e., epochs) until gradient descent's convergence.

        The property tolerance denotes the tolerated difference between the current loss and previous loss.
        If the magnitude of the gradient falls below the tolerance level, then gradient descent has converged.
        Thus, similar to the maximum number of iterations, the tolerance determines the convergence of gradient descent.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance

    def func(self, x):
        """
        Loss computation after each iteration.
        """
        f_x = np.exp(-x/2) + 10*np.power(x,2) 
        return f_x

    def f_prime(self, x):
        """
        Compute derivative of f_(x). After each iteration, the derivative of f(x) is multiplied by the learning rate alpha
        and subtracted from the current gradient.
        """
        d_x = -0.5 * np.exp(-x/2) + 20*x
        return d_x 

    def step(self, d_x):
        """
        Step after each iteration. Derivative of f_(x) is multiplied by the learning rate alpha.
        """
        alpha = self.alpha
        return alpha * d_x 

    def computation(self, x: int, convergence = False):
        """
        This function computes the actual gradients and is the core function of the gradient descent algorithm.
        x is an integer which denotes the starting point. x is initialized with 1 (i.e., x_0 = 1).

        Gradients (d_x) and function values (f_(x)) are simultaneously calculated until convergence or max num of iterations. 
        """

        # max number of iterations / epochs until convergence
        max_iter = self.max_iter

        # learning rate
        alpha = self.alpha

        num_iter = 1

        if convergence == True:

            y = self.func(x)
            f_vals = [y]
            
            # determine tolerance (by default set to 1e-10)
            tolerance = self.tolerance
            convergence = 0

            while convergence < 1:
     
                d_x = self.f_prime(x)

                # take step in the direction of steepest descent
                x -= self.step(d_x)

                y_current = self.func(x)
                f_vals.append(y_current)

                num_iter += 1 

                diff = abs(y - y_current)
                y = y_current

                # if magnitude of the gradient < tolerance level, then gradient descent has converged
                if diff < tolerance:
                    convergence = 1

                # if number of iterations > maximum number of iterations, then gradient descent has converged
                elif num_iter > max_iter:
                    convergence = 1 

            return f_vals, num_iter

        else:
            
            plot_range = 5
            plot_x1 = -plot_range
            plot_x2 =  plot_range

            tangents = []
            step_sizes = []

            for _ in range(max_iter):

                y = self.func(x)

                d_x = self.f_prime(x)

                step_size = self.step(d_x)

                if num_iter <= 3:
                       
                    y_int = (-d_x * x) + y
                    y_neg_xrange = (-d_x * plot_x1) + y_int
                    y_pos_xrange = (d_x * plot_x2) + y_int
                    tangents.append(([plot_x1, x, plot_x2], [-y_neg_xrange, y, y_pos_xrange]))
                
                if num_iter <= 10:

                    step_sizes.append(step_size)

                # take step in the direction of steepest descent
                x -= step_size

                num_iter += 1

            x = np.arange(plot_x1, plot_x2, 0.001)
            y = self.func(x)

            # plot tangent lines if and only if maximum number of iterations is equal to 3
            if max_iter == 3:
                
                # plot function itself
                plt.plot(x, y)

                for tangent in tangents:
                    # plot tangent lines
                    plt.plot(tangent[0], tangent[1])
                    plt.plot(tangent[0][1], tangent[1][1], 'o')
                
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(r'Gradient descent over %d iterations ($\alpha = %s $)' % (max_iter, alpha))
                plt.show()

            # if maximum number of iterations is equal to 10, do not plot tangent lines (only plot steps)
            if max_iter == 10:

                self.plot_steps(step_sizes)

                
    def plot_steps(self, step_sizes: list):
        """
        Function to visualize the steps of gradient descent.
        """

        alpha = self.alpha
        max_iter = self.max_iter

        plt.plot(range(1,max_iter+1), step_sizes, '-o')
        plt.xlabel("Number of iterations")
        plt.ylabel("Step size")
        plt.title(r'Gradient descent steps over %d iterations ($\alpha = %s $)' % (max_iter, alpha))
        plt.show()

    def plot_losses(self, losses: list):
        """
        Function to visualize the loss (minimization) over time. Needs a list of losses as input argument.
        """
        alpha = self.alpha

        plt.plot(losses)
        plt.xlabel("Number of iterations")
        plt.ylabel("f_(x)")
        plt.title(r'Gradient descent until convergence ($\alpha = %s $)' % alpha)
        plt.show()


class PCA():

    # create initializer method
    def __init__(self, n_dims: int, scaling: str):
        """
        The property 'n_dims' determines the number of principal components the data set should be projected onto.

        The property 'scaling' determines whether normalization or mean centering should be computed.
        For the latter property, there is no other option than normalization / standardization or mean centering. 
        """
        self.n_dims = n_dims
        self.scaling = scaling

    @staticmethod
    def p_components(X):
        """
        Use Bessel's correction and divide by n - 1.
        Return evals in descending order.
        Sort eigenvectors (i.e., columns of V) according to its corresponding eigenvalues.

        For indexing, eigenvector matrix V is required to be transposed,
        as each column of V[:,i] is the eigenvector corresponding to the eigenval eval[i].

        P acts as a generalized rotation to align a basis with the axis of maximal variance,
        resulting ordered set of p’s are the principal components (each row of P (i.e., p_i) is an eigenvector of cov matrix C_X).
      
        """

        n_samples = X.shape[1]

        C_X = X.dot(X.T) / (n_samples - 1)
        evals, V = np.linalg.eig(C_X)

        index_descending = np.argsort(evals)[::-1]

        evals_desc = evals[index_descending]
        evecs_desc = V.T[index_descending]

        P = evecs_desc
        Y = P.dot(X)

        C_Y = Y.dot(Y.T) / Y.shape[1]

        #variances of input data along each principal component (i.e., p_i) / eigenvalues of C_Y
        var = [col for i, row in enumerate(C_Y) for j, col in enumerate(row) if i == j]

        return P, var

    def mds(self, X):
        """
        Multidimensional scaling. This function computes the dimensionality reduction. 
        Center data (important for PCA), normalization is not necessary.
        """
        if self.scaling == "mean_centering":
            scaler = Scaler(method ='centering')
            X = scaler.normalization(X).T
        elif self.scaling == "normalization":
            scaler = Scaler(method ='normalization')
            X = scaler.normalization(X).T
        else:
            raise Exception("This argument can only be set to 'mean_centering' or 'normalization'.")

        # P = rotation matrix, where each row corresponds to a principal component of X
        P, evals = PCA.p_components(X)

        Y = P.dot(X)

        d = self.n_dims

        PC_d = Y.T[:,:d]

        return PC_d

    def cum_var(self, evals):
        """
        Compute cumulative variance.
        """
        c_var = np.cumsum(evals/np.sum(evals))
        return c_var