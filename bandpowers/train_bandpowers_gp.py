'''
Author: Arrykrishna Mootoovaloo

The GPs are trained in parallel via the function below.

Note that we supply a table in the following format:

|inputParameters|B_0|B_1|B_2|B_3|...|B_23|

where:
- inputParameters is a matrix of size N_train and N_dim
- B_0, B_1, ..., B_23 is the i^th (transformed) band power evaluated at these input parameters
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
    inputs:
        directory (string):
    '''

    def __init__(
            self,
            directory,
            n_band=24,
            file_name='training_points/bandpowers_1000.csv',
            savedir='gps/bandpowers',
            ndim=8):
        self.n_band = n_band
        self.ndim = ndim
        self.savedir = savedir
        self.all_data = []
        self.arguments = []
        self.directory = directory
        self.file_name = file_name
        self.data = np.array(pd.read_csv(self.directory + self.file_name))[:, 1:]
        self.input = self.data[:, 0:self.ndim]

        if not os.path.exists(self.directory + self.savedir):
            os.makedirs(self.directory + self.savedir)

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
                direct = self.directory + self.savedir + '/' + file_name + '_' + str((i * n_process) + j)
                with open(direct + '.pkl', 'wb') as gp_dummy:
                    dill.dump(gps[j], gp_dummy)

            del gps

if __name__ == '__main__':

# Option-4/ : 1000_8D_table_prior_moped_3_998.csv
# Option-5/ : 1500_8D_table_prior_moped_3_1499.csv
# Option-6/ : 2000_8D_table_prior_moped_3_1998.csv
# Option-7/ : 2500_8D_table_prior_moped_3_2499.csv
# Option-8/ : 3000_8D_table_prior_moped_3_3000.csv

    DIR = '/Users/Harry/Dropbox/gp_emulator/'

    SIGMA = [-40.0]
    TRAIN = True

    GPS = TrainGP(
        directory=DIR,
        n_band=24,
        file_name='training_points/bandpowers_1000.csv',
        savedir='gps/bandpowers',
        ndim=8)
    GPS.generate_args(sigma=SIGMA, training=TRAIN, nrestart=5)
    # GPS.trainings(n_process = 8, file_name = 'gp')
    