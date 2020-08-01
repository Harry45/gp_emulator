'''
Authors: Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
Affiliation: Imperial College London
Department: Imperial Centre for Inference and Cosmology
Email: a.mootoovaloo17@imperial.ac.uk

Description:
Gaussian Process script for training Gaussian Processes

This code has been tested in Python 3 version 3.7.4
'''

import os
import numpy as np
import dill
from sklearn.cluster import KMeans
import sklearn.preprocessing as skp
from multiprocessing import Pool
import pandas as pd
from gp_moped import GP

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=True)

bnd = np.repeat(np.array([[-1.5, 6]]), 9, axis=0)
bnd[0] = np.array([-1.5, 2])


def worker(args):
    myGP = GP(*args)
    myGP.transform()
    myGP.fit(method='L-BFGS-B', bounds=bnd, options={'ftol': 1E-12, 'maxiter': 500})
    return myGP


class training:
    '''
    Inputs
    ------
    n_moped (int) : number of MOPED coefficients

    file_name (str) : point to the training set

    savedir (str) : directory where we want to save the GPs

    ndim (int) : dimensionality of the problem
    '''

    def __init__(self, n_moped=11, file_name='training_points/moped_coeffs_1000.csv', savedir='gps', ndim=8):

        # number of MOPED coefficients
        self.n_moped = n_moped

        # number of dimensions
        self.ndim = ndim

        # directory where we want to save the GPs
        self.savedir = savedir

        # list to store inputs to the worker function
        self.arguments = []

        # file name for the GP
        self.file_name = file_name

        # the data file
        self.data = np.array(pd.read_csv(self.file_name))[:, 1:]

        # inputs to the GP (first ndim column)
        self.input = self.data[:, 0:self.ndim]

        # create directory where we want to store the GPs
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def gen_arguments(self, sigma=[-4.0], train=False, nrestart=1):
        '''
        Function to generate arguments that will be fed into the function (worker),
        which will allow us to train the GPs in parallel

        Inputs
        ------
        sigma (np.ndarray) : the noise/jitter term for the Gaussian Process

        train (bool) : if True, the GPs will be trained

        nrestart (int) : number of times we want to restart training

        Returns
        -------

        arguments (list) : list of all the inputs to the worker function
        '''

        for k in range(self.n_moped):

            input_data = np.c_[self.input, self.data[:, k + self.ndim]]
            arguments.append([input_data, sigma, train, nrestart, self.ndim])

        return self.arguments

    def trainings(self, n_proc=8, file_name='gp'):
        '''
        Function to train the GPs in parallel

        Inputs
        ------

        n_proc (int) : number of processors

        file_name (str) : file name for the GP

        Returns
        -------
        '''

        for i in range(2):

            if i == 0:

                pool = Pool()
                gps = pool.map(worker, self.arguments[0:8])
                pool.close()

                for j in range(n_proc):

                    with open(self.savedir + '/' + file_name + '_' + str(j) + '.pkl', 'wb') as f:
                        dill.dump(gps[j], f)

                del gps

            else:

                pool = Pool()
                gps = pool.map(worker, self.arguments[8:])
                pool.close()

                for j in range(int(self.n_moped - n_proc)):

                    with open(self.savedir + '/' + file_name + '_' + str(j + 8) + '.pkl', 'wb') as f:
                        dill.dump(gps[j], f)

                del gps


if __name__ == '__main__':

    sigma = [-40.0]
    train = True

    trainGPs = training(n_moped=11, file_name='training_points/moped_coeffs_1000.csv', savedir='gps')
    trainGPs.gen_arguments(sigma=sigma, train=True, nrestart=5)
    # trainGPs.trainings(n_proc = 8, file_name = 'gp')
