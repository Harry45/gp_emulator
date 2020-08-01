'''
Author: Arrykrishna Mootoovaloo
Email: a.mootoovaloo17@imperial.ac.uk
Status: Under development
Description: script for training the GPs for the band powers
'''

import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import dill
from gp_bandpowers import GP

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=True)

# Bounds (prior) fo training the GPs
BND = np.repeat(np.array([[-1.5, 6]]), 9, axis=0)
BND[0] = np.array([-1.5, 2])


def worker(args):
    '''
    inputs to the GP class
    '''
    my_gp = GP(*args)
    my_gp.transform()
    my_gp.fit(method='L-BFGS-B', bounds=BND, options={'ftol': 1E-12, 'maxiter': 500})
    return my_gp


class TrainGP:
    '''
    inputs
    ------
    n_band (int) : number of band powers

    file_name (str) : the csv file containing the training points

    savedir (str) : directory for storing the GPs

    ndim (int) : the number of dimension for the problem

    '''

    def __init__(self, n_band=24, file_name='training_points/bandpowers_1000.csv', savedir='gps', ndim=8):

        # number of band powers
        self.n_band = n_band

        # number of dimension
        self.ndim = ndim

        # directory where we want to save the GPs
        self.savedir = savedir

        # list to store the inputs to the GP
        self.arguments = []

        # the file name for the training points
        self.file_name = file_name

        # the data
        self.data = np.array(pd.read_csv(self.file_name))[:, 1:]

        # inputs to the emulator
        self.input = self.data[:, 0:self.ndim]

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def generate_args(self, sigma, training=False, nrestart=1):
        '''
        inputs:
            sigma (np.ndarray): the noise term - we assume it's just the
                jitter term

            training (bool); if true, GPs will be trained

            nrestart (int): number of times we want to restart the optimier

        outputs:
            None
        '''

        for k in range(self.n_band):

            input_data = np.c_[self.input, self.data[:, k + self.ndim]]
            self.arguments.append([input_data, sigma, training, nrestart, self.ndim])

    def trainings(self, n_process=8, file_name='gp'):
        '''
        inputs:
                n_process = 8
                file_name = 'gp'

        DEFAULT:
                n_process should be 8 because training 24 (8 at a time in parallel)
        '''

        step = int(np.floor(self.n_band / n_process))

        for i in range(step):

            pool = Pool()
            gps = pool.map(worker, self.arguments[i * n_process:(i + 1) * n_process])
            pool.close()

            for j in range(n_process):
                direct = self.savedir + '/' + file_name + '_' + str((i * n_process) + j)
                with open(direct + '.pkl', 'wb') as gp_dummy:
                    dill.dump(gps[j], gp_dummy)

            del gps


if __name__ == '__main__':

    SIGMA = [-40.0]
    TRAIN = True

    GPS = TrainGP(n_band=24, file_name='training_points/bandpowers_1000.csv', savedir='gps', ndim=8)
    GPS.generate_args(sigma=SIGMA, training=TRAIN, nrestart=5)
    # GPS.trainings(n_process = 8, file_name = 'gp')
