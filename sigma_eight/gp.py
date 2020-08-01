'''
Author: Arrykrishna Mootoovaloo
Email: a.mootoovaloo17@imperial.ac.uk
Status: Under development
Description: Gaussian Process Scipt for sigma_8
'''
import logging
import os
import numpy as np
import scipy.optimize as op
from scipy.spatial.distance import cdist
from GPy.util import linalg as gpl


# ignore some numerical errors
# and print floats at a certain precision
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=False)

# Settings for our log file
FORMAT = "%(levelname)s:%(filename)s.%(funcName)s():%(lineno)-8s %(message)s"
FILENAME = "logs/gaussian_process_sigma_eight.log"

def distanceperdim(x_1, x_2):
    '''
    Function to compute pairwise distance for each dimension

    Args:
        (array) x_1: a vector of length equal to the number
                of training points

        (array) x_2: a vector of length equal to the number
                of training points

    Returns:
        (array) a matrix of size N x N
    '''

    # reshape first and second vector in
    # the right format
    x_1 = x_1.reshape(len(x_1), 1)
    x_2 = x_2.reshape(len(x_2), 1)

    # compute pairwise squared euclidean distance
    dist_sq = cdist(x_1, x_2, metric='sqeuclidean')

    return dist_sq

class GP:
    '''
    GP class
    '''
    def __init__(self, data, sigma, train=False, nrestart=5):
        '''GP class

        :param data (np.ndarray): size N x ndim (dimension of the problem) + 1.
                The first ndim columns contain the inputs to the GP and the
                last column contains the output

        :param sigma (np.ndarray): size N or 1. We assume noise-free
                regression. default: [-5.0] and this is log-standard
                deviation. This is also referred to as the jitter term
                in GP for numerical stability. At this point, the code
                does not support full noise covariance matrix.

        :param train (bool): True indicates that we will train the
                GP, otherwise, it uses the default values of the
                kernel parameters

        :param nrestart (int): Number of times we want to restart the optimiser

        :param ndim (int): the dimension of the problem
        '''

        if not os.path.exists('logs'):
            os.mkdir('logs')

        should_roll_over = os.path.isfile(FILENAME)
        handler = logging.handlers.RotatingFileHandler(
            FILENAME, mode='w', backupCount=0)

        # log already exists, roll over
        if should_roll_over:
            handler.doRollover()

        # Create our log file
        logging.basicConfig(
            filename=FILENAME, level=logging.DEBUG, format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initialising variables...')

        # data
        data = np.array(data)

        # inputs
        self.theta = data[:, 0:-1]

        # outputs
        self.output_original = data[:, -1]

        # compute mean of output
        self.mean_y = np.mean(self.output_original)

        # compute standard deviation of output
        self.std_y = np.std(self.output_original)

        # standardize output
        self.output = (self.output_original - self.mean_y) / self.std_y

        # number of training points
        self.ntrain = len(self.output)

        # jitter term
        self.sigma = np.array(sigma)

        # boolean - training
        self.train = train

        # number of restart
        self.nrestart = nrestart

        # dimension of the problem
        self.ndim = self.theta.shape[1]

        # record width
        self.width = None

        # record scale
        self.scale = None

        # empty list to record the cost
        self.minchi_sq = []

        # empty list to record the optimum parameters
        self.recordparams = []

        # record alpha
        self.alpha_ = None

        # record transformed inputs
        self.theta_ = None

        # record Cholesky factor
        self.chol_fact = None

        # record transformation matrix
        self.mu_matrix = None

    def transform(self):
        '''
        Function to pre-whiten the input parameters

        Args:
            None: uses the inputs to the class method above

        Returns:
            None: we store the transformation matrix
            and the transformed parameters
        '''

        # first compute the covariance matrix of the inputs
        cov = np.cov(self.theta.T)

        # then compute the SVd of the covariance matrix
        phi, eigvals, phi_t = np.linalg.svd(cov)

        del phi

        # Compute 1/square-root of the eigenvalues
        eigvals_block = np.diag(1.0 / np.sqrt(eigvals))

        # compute and store the transformation matrix
        self.mu_matrix = np.dot(eigvals_block, phi_t)

        self.logger.info(
            'Size of the transformation matrix is {}'.format(self.mu_matrix.shape))

        # compute the transformed parameters
        self.theta_ = np.dot(self.mu_matrix, self.theta.T)

        # transpose - to get the N x d matrix back
        self.theta_ = self.theta_.T

    def rbf(self, label, x_1, x_2=None):
        '''
        Function to generate the RBF kernel

        Args:
            (str) label: 'trainSet', 'trainTest' and 'testSet'
                        same notations as in Rasmussen (2006)

            (array) x_1: first N x d inputs

            (array) x_2: second set of N x d array - can
                        either be training set or test point

        Returns:
            (array) : either k or k_s or k_ss

        '''

        # Amplitude of kernel
        amp = np.exp(2.0 * self.width)

        # divide inputs by respective characteristic lengthscales
        x_1 = x_1 / np.exp(self.scale)
        x_2 = x_2 / np.exp(self.scale)

        # Compute pairwise squared euclidean distance
        distance = cdist(x_1, x_2, metric='sqeuclidean')

        # Generate kernel or vector (if test point)
        if label == 'trainSet':
            k = amp * np.exp(-0.5 * distance)

        elif label == 'trainTest':
            k = amp * np.exp(-0.5 * distance)
            k = k.flatten()
            k = k.reshape(len(k), 1)
        else:
            k = amp.reshape(1, 1)

        return k

    def kernel(self, label, x_1, x_2=None):
        '''
        Function to compute the kernel matrix

        Args:
            (str) label: 'trainSet', 'trainTest' and 'testSet'
                        same notations as in Rasmussen (2006)

            (array) x_1: first N x d inputs

            (array) x_2: second set of N x d array - can
                        either be training set or test point

        Returns:
            (array) : either k or k_s or k_ss

        '''

        if label == 'trainSet':
            k = self.rbf('trainSet', x_1, x_2)
            np.fill_diagonal(k, k.diagonal() + np.exp(2.0 * self.sigma))
        else:
            k = self.rbf(label, x_1, x_2)
        return k

    def alpha(self):
        '''
        Function to compute alpha = k^-1 y

        Args:
            None

        Returns:
            (array) alpha of size N x 1
        '''

        # compute the kernel matrix of size N x N
        k = self.kernel('trainSet', self.theta_, self.theta_)

        # compute the Cholesky factor
        self.chol_fact = gpl.jitchol(k)

        # Use triangular method to solve for alpha
        alp = gpl.dpotrs(self.chol_fact, self.output, lower=True)[0]

        return alp

    def cost(self, theta):
        '''
        Function to calculate the negative log-marginal likelihood
        (cost) of the Gaussian Process

        Args:
            (array) theta: the kernel hyperparameters

        Returns:
            (array) cost: outputs the cost (1x_1 array)
        '''

        # Sometimes L-BFGS-B was crazy - flattening the vector
        theta = theta.flatten()

        # first element is the amplitude
        self.width = theta[0]

        # the remaining elements are the characteristic lengthscales
        self.scale = theta[1:]

        # compute alpha
        alpha_ = self.alpha()

        # trick to compute the determinant once we have
        # already computed the Cholesky factor
        det_ = np.log(np.diag(self.chol_fact)).sum(0)

        # compute the cost
        cst = 0.5 * (self.output * alpha_).sum(0) + det_

        return cst

    def grad_log_like(self, theta):
        '''
        Function to calculate the gradient of the cost
        (negative log-marginal likelihood) with respect to
        the kernel hyperparameters

        Args:
            (array) theta: the kernel hyperparameters in
                            the correct order

        Returns:
            (array) gradient: vector of the gradient
        '''

        # the kernel hyperparameters
        theta = theta.flatten()

        # amplitude
        self.width = theta[0]

        # characteristic lengthscales
        self.scale = theta[1:]

        # Number of parameters
        n_params = len(theta)

        # empty array to record the gradient
        gradient = np.zeros(n_params)

        # compute alpha
        alpha_ = self.alpha()

        # compute k^-1 via triangular method
        kinv = gpl.dpotrs(self.chol_fact, np.eye(self.ntrain), lower=True)[0]

        # see expression for gradient
        dummy = np.einsum('i,j', alpha_.flatten(), alpha_.flatten()) - kinv

        # Gradient calculation with respect
        # to hyperparameters (hard-coded)
        grad = {}
        k_rbf = self.rbf('trainSet', self.theta_, self.theta_)

        grad['0'] = 2.0 * k_rbf
        for i in range(self.ndim):
            dist_ = distanceperdim(self.theta_[:, i], self.theta_[:, i])
            grad[str(i + 1)] = k_rbf * dist_ / np.exp(2.0 * self.scale[i])

        for i in range(n_params):
            gradient[i] = 0.5 * gpl.trace_dot(dummy, grad[str(i)])

        return -gradient

    def fit(self, method, bounds, options):
        '''
        Function to do the optimisation (training)

        Args:
            (str) method: optimisation method from scipy
                    see scipy.optimize.minimize for further details

            (array) bounds: some methods also allow for a bound/prior

            (dict) options: can also pass convergence conditions etc...
        '''

        bounds_ = np.array(bounds)

        # if we want to train
        if self.train:

            for i in range(self.nrestart):

                print('Performing Optimization step {}'.format(i + 1))

                # an initial guess from the bound/prior
                guess = np.random.uniform(bounds_[:, 0], bounds_[:, 1])

                # optimisation!
                soln = op.minimize(
                    self.cost, guess, method=method,
                    bounds=bounds, jac=self.grad_log_like, options=options)

                # record optimised solution (cost)
                self.minchi_sq.append(np.ones(1) * soln.fun)

                # record optimum parameters
                self.recordparams.append(soln.x)

            # just converting list to arrays
            self.minchi_sq = np.array(self.minchi_sq).reshape(self.nrestart,)
            self.recordparams = np.array(self.recordparams)

            # sometimes we found crazy behaviour
            # maybe due to numerical errors
            # ignore NaN in cost
            if np.isnan(self.minchi_sq).any():
                index = np.argwhere(np.isnan(self.minchi_sq))
                self.minchi_sq = np.delete(self.minchi_sq, index)
                self.recordparams = np.delete(self.recordparams, index, axis=0)

            self.logger.info(
                'The value of the cost is {}'.format(self.minchi_sq))

            # choose the hyperparameters with minimum cost
            cond = self.minchi_sq == np.min(self.minchi_sq)
            opt_params = self.recordparams[cond][0]
            opt_params = opt_params.flatten()

            self.logger.info('Optimum is {}'.format(opt_params))

            # update amplitude
            self.width = opt_params[0]

            # update characteristic lengthscales
            self.scale = opt_params[1:]

            # Update alpha (after training) - no need to update kernel
            # because already updated in function cost
            self.alpha_ = self.alpha()

        else:
            self.alpha_ = self.alpha()

    def prediction(self, testpoint, returnvar=True):
        '''
        Function to make predictions given a test point

        Args:
            (array) testpoint: a test point of length ndim

            (bool) returnvar: If True, the GP variance will
                    be computed

        Returns:
            (array) mean, var: if returnvar=True
            (array) mean : if returnvar=False
        '''

        # use numpy array instead of list (if any)
        testpoint = np.array(testpoint).flatten()

        assert len(testpoint) == self.ndim, 'different dimension'

        # transform point first
        testpoint_trans = np.dot(self.mu_matrix, testpoint)
        testpoint_trans = testpoint_trans.reshape(1, self.ndim)

        # compute the k_star vector
        k_s = self.kernel('trainTest', self.theta_, testpoint_trans)

        # compute mean GP - super quick
        mean_gp = np.array([(k_s.flatten() * self.alpha_.flatten()).sum(0)])

        # rescale back
        mean_scaled = self.mean_y + self.std_y * mean_gp

        # do extra computations if we want GP variance
        if returnvar:
            variance = gpl.dpotrs(self.chol_fact, k_s, lower=True)[0].flatten()
            k_ss = self.kernel('testSet', testpoint_trans, testpoint_trans)
            var_gp = k_ss - (k_s.flatten() * variance).sum(0)
            var_gp = var_gp.flatten()

            # rescale back
            var = self.std_y**2 * var_gp
            return mean_scaled, var

        return mean_scaled
