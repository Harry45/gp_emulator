'''
Author: Arrykrishna Mootoovaloo
Code adapted from original KiDS-450 release Python Code

Most of this code is analogous the
the original likelihood code except that
we use GPs for doing the inference
'''


import imp
import os
import warnings
import time
from collections import OrderedDict
import numpy as np
import scipy.interpolate as itp
from scipy.linalg import cholesky, solve_triangular, expm
from scipy.special import logsumexp
import dill
import emcee
from GPy.util import linalg as gpl
from utils.genDistribution import genDist

# ignore warnings
warnings.filterwarnings("ignore")

# some additional setups
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, formatter={'float_kind': '{:0.2f}'.format})


class EMULATOR:
    '''
    Emulator CLASS

    param directory (str): current working directory

    param file_settings (str): your setting file

    param set_random (bool): if true, samples of GPs will be used

    param n_realisation (int): the number of GP samples which will be used
    '''

    def __init__(self, file_settings='settings', set_random=False, n_realisation=10):

        self.n_realisation = n_realisation
        self.set_random = set_random
        self.file_settings = file_settings
        self.settings = imp.load_source(
            self.file_settings, self.file_settings)

        self.redshift_bins = []

        for index_zbin in range(len(self.settings.zbin_min)):
            min_z = self.settings.zbin_min[index_zbin]
            max_z = self.settings.zbin_max[index_zbin]
            redshift_bin = '{:.2f}z{:.2f}'.format(min_z, max_z)
            self.redshift_bins.append(redshift_bin)

        # number of z-bins
        self.nzbins = len(self.redshift_bins)
        # number of *unique* correlations between z-bins
        self.nzcorrs = int(self.nzbins * (self.nzbins + 1) / 2)

        self.all_bands_ee_to_use = []
        self.all_bands_bb_to_use = []

        # default, use all correlations:
        for _ in range(self.nzcorrs):
            self.all_bands_ee_to_use += self.settings.bands_EE_to_use
            self.all_bands_bb_to_use += self.settings.bands_BB_to_use

        self.all_bands_ee_to_use = np.array(self.all_bands_ee_to_use)
        self.all_bands_bb_to_use = np.array(self.all_bands_bb_to_use)

        all_bands_to_use = np.concatenate((self.all_bands_ee_to_use,
                                           self.all_bands_bb_to_use))
        self.indices_for_bands_to_use = np.where(
            np.asarray(all_bands_to_use) == 1)[0]
        # print(self.indices_for_bands_to_use)

        # this is also the number of points in the datavector
        ndata = len(self.indices_for_bands_to_use)

        if self.settings.correct_resetting_bias:
            fname = os.path.join(
                self.settings.data_directory,
                'Resetting_bias/parameters_B_mode_model.dat')
            a_b_modes, exp_b_modes, err_a_b_modes, err_exp_b_modes = np.loadtxt(
                fname, unpack=True)
            del err_a_b_modes
            del err_exp_b_modes
            self.params_resetting_bias = np.array([a_b_modes, exp_b_modes])

            fname = os.path.join(
                self.settings.data_directory,
                'Resetting_bias/covariance_B_mode_model.dat')
            self.cov_resetting_bias = np.loadtxt(fname)

        # try to load fiducial m-corrections from file
        # (currently these are global values over full field,
        # hence no looping over fields required for that!)
        # to do : Make output dependent on field, not necessary for current KiDS
        # approach though!
        try:
            fname = os.path.join(self.settings.data_directory,
                                 '{:}zbins/m_correction_avg.txt'.format(self.nzbins))

            if self.nzbins == 1:
                self.m_corr_fiducial_per_zbin = np.asarray(
                    [np.loadtxt(fname, usecols=[1])])
            else:
                self.m_corr_fiducial_per_zbin = np.loadtxt(fname, usecols=[1])
        except BaseException:
            self.m_corr_fiducial_per_zbin = np.zeros(self.nzbins)
            print('Could not load m-correction values from {}\n'.format(fname))
            print('Setting them to zero instead.')

        try:
            fname = os.path.join(
                self.settings.data_directory,
                '{:}zbins/sigma_int_n_eff_{:}zbins.dat'.format(
                    self.nzbins,
                    self.nzbins))
            tbdata = np.loadtxt(fname)
            if self.nzbins == 1:
                # correct columns for file!
                sigma_e1 = np.asarray([tbdata[2]])
                sigma_e2 = np.asarray([tbdata[3]])
                n_eff = np.asarray([tbdata[4]])
            else:
                # correct columns for file!
                sigma_e1 = tbdata[:, 2]
                sigma_e2 = tbdata[:, 3]
                n_eff = tbdata[:, 4]

            self.sigma_e = np.sqrt((sigma_e1**2 + sigma_e2**2) / 2.)
            # convert from 1 / sq. arcmin to 1 / sterad
            self.n_eff = n_eff / np.deg2rad(1. / 60.)**2
        except BaseException:
            # these dummies will set noise power always to 0!
            self.sigma_e = np.zeros(self.nzbins)
            self.n_eff = np.ones(self.nzbins)
            print('Could not load sigma_e and n_eff!')

        collect_bp_ee_in_zbins = []
        collect_bp_bb_in_zbins = []
        # collect BP per zbin and combine into one array
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):
                # zbin2 first in fname!
                fname_ee = os.path.join(
                    self.settings.data_directory,
                    '{:}zbins/band_powers_EE_z{:}xz{:}.dat'.format(
                        self.nzbins,
                        zbin1 + 1,
                        zbin2 + 1))
                fname_bb = os.path.join(
                    self.settings.data_directory,
                    '{:}zbins/band_powers_BB_z{:}xz{:}.dat'.format(
                        self.nzbins,
                        zbin1 + 1,
                        zbin2 + 1))
                extracted_bp_ee = np.loadtxt(fname_ee)
                extracted_bp_bb = np.loadtxt(fname_bb)
                collect_bp_ee_in_zbins.append(extracted_bp_ee)
                collect_bp_bb_in_zbins.append(extracted_bp_bb)

        self.band_powers = np.concatenate(
            (np.asarray(collect_bp_ee_in_zbins).flatten(),
             np.asarray(collect_bp_bb_in_zbins).flatten()))

        self.n_band = len(np.asarray(collect_bp_ee_in_zbins).flatten())

        fname = os.path.join(self.settings.data_directory,
                             '{:}zbins/covariance_all_z_EE_BB.dat'.format(self.nzbins))
        self.covariance = np.loadtxt(fname)

        fname = os.path.join(self.settings.data_directory,
                             '{:}zbins/band_window_matrix_nell100.dat'.format(self.nzbins))
        self.band_window_matrix = np.loadtxt(fname)
        # ells_intp and also band_offset are consistent between different
        # patches!

        fname = os.path.join(
            self.settings.data_directory,
            '{:}zbins/multipole_nodes_for_band_window_functions_nell100.dat'.format(
                self.nzbins))
        self.ells_intp = np.loadtxt(fname)

        self.band_offset_ee = len(extracted_bp_ee)
        self.band_offset_bb = len(extracted_bp_bb)

        # Read fiducial dn_dz from window files:
        # to do : the hardcoded z_min and z_max correspond to the lower and upper
        # endpoints of the shifted left-border histogram!
        z_samples = []
        hist_samples = []
        for zbin in range(self.nzbins):
            redshift_bin = self.redshift_bins[zbin]
            window_file_path = os.path.join(
                self.settings.data_directory,
                '{:}/n_z_avg_{:}.hist'.format(
                    self.settings.photoz_method,
                    redshift_bin))
            zptemp, hist_p_redshift = np.loadtxt(
                window_file_path, usecols=[
                    0, 1], unpack=True)
            shift_to_midpoint = np.diff(zptemp)[0] / 2.
            # we add a zero as first element because we want to integrate down
            # to z = 0!
            z_samples += [np.concatenate((np.zeros(1),
                                          zptemp + shift_to_midpoint))]
            hist_samples += [np.concatenate((np.zeros(1), hist_p_redshift))]

        z_samples = np.asarray(z_samples)
        hist_samples = np.asarray(hist_samples)

        # prevent undersampling of histograms!
        if self.settings.nzmax < len(zptemp):
            cmnt = "You're trying to integrate at lower resolution than " + \
                "supplied by the n(z) histograms. " + "\n Increase nzmax! Aborting now..."
            print(cmnt)
        # if that's the case, we want to integrate at histogram resolution and need to account for
        # the extra zero entry added
        elif self.settings.nzmax == len(zptemp):
            self.settings.nzmax = z_samples.shape[1]
            # requires that z-spacing is always the same for all bins...
            self.redshifts = z_samples[0, :]
            print('Integrations performed at resolution of histogram!')
            # if we interpolate anyway at arbitrary resolution the extra 0
            # doesn't matter
        else:
            self.settings.nzmax += 1
            self.redshifts = np.linspace(
                z_samples.min(), z_samples.max(), self.settings.nzmax)
            print('Integration performed at set nzmax resolution!')

        self.p_redshift = np.zeros((self.settings.nzmax, self.nzbins))
        self.p_redshift_norm = np.zeros(self.nzbins, 'float64')

        for zbin in range(self.nzbins):
            # we assume that the histograms loaded are given as left-border histograms
            # and that the z-spacing is the same for each histogram
            spline_p_redshift = itp.splrep(z_samples[zbin, :], hist_samples[zbin, :])

            # z_mod = self.z_p
            mask_min = self.redshifts >= z_samples[zbin, :].min()
            mask_max = self.redshifts <= z_samples[zbin, :].max()
            mask = mask_min & mask_max
            # points outside the z-range of the histograms are set to 0!
            self.p_redshift[mask, zbin] = itp.splev(self.redshifts[mask], spline_p_redshift)
            # Normalize selection functions
            dz = self.redshifts[1:] - self.redshifts[:-1]
            self.p_redshift_norm[zbin] = np.sum(
                0.5 * (self.p_redshift[1:, zbin] + self.p_redshift[:-1, zbin]) * dz)

        self.z_max = self.redshifts.max()

        if self.settings.mode == 'halofit':
            self.class_argumets = {
                'z_max_pk': self.z_max,
                'output': 'mPk',
                'non linear': self.settings.mode,
                'P_k_max_h/Mpc': self.settings.k_max_h_by_Mpc}
        else:
            self.class_argumets = {
                'z_max_pk': self.z_max,
                'output': 'mPk',
                'P_k_max_h/Mpc': self.settings.k_max_h_by_Mpc}

        fname = os.path.join(
            self.settings.data_directory,
            'number_datapoints.txt')
        np.savetxt(
            fname,
            [ndata],
            header='number of datapoints in masked datavector')

        # 1) determine l-range for taking the sum, #l = l_high-l_min at least!!!:
        # this is the correct calculation!
        # for real data, I should start sum from physical scales, i.e., currently l>= 80!
        # to do : Set this automatically!!!
        # Not automatically yet, but controllable via "myCFHTLenS_tomography.data"!!!
        # these are integer l-values over which we will take the sum used in
        # the convolution with the band window matrix
        self.ells_min = self.ells_intp[0]
        self.ells_max = self.ells_intp[-1]
        self.nells = int(self.ells_max - self.ells_min + 1)
        self.ells_sum = np.linspace(self.ells_min, self.ells_max, self.nells)

        # these are the l-nodes for the derivation of the theoretical Cl:
        self.ells = np.logspace(
            np.log10(
                self.ells_min), np.log10(self.ells_max), self.settings.nellsmax)
        self.ell_norm = self.ells_sum * (self.ells_sum + 1) / (2. * np.pi)

        self.band_ee_selected = np.tile(
            self.settings.bands_EE_to_use, self.nzcorrs)
        self.bands_bb_selected = np.tile(
            self.settings.bands_BB_to_use, self.nzcorrs)

        # Account for m-correction if it is fixed to fiducial values

        if 'm_corr' not in self.settings.use_nuisance:
            m_m, m_c = self.calc_m_correction()
            self.covariance = self.covariance / np.asarray(m_c)
            self.covariance = self.covariance[np.ix_(
                self.indices_for_bands_to_use, self.indices_for_bands_to_use)]
            self.band_powers = self.band_powers / np.asarray(m_m)
            self.band_powers = self.band_powers[self.indices_for_bands_to_use]
            self.cov_inverse = np.linalg.inv(self.covariance)
            self.chol_fact = cholesky(self.covariance, lower=True)

    def calc_m_correction(self):

        '''
        Calculates the m correction

        inputs:
            None

        outputs:
            m_correction (ndarray)

            m_correction matrix (ndarray)
        '''

        param_name = 'm_corr'
        if param_name in self.settings.use_nuisance:
            m_corr = self.systematics['m']
            delta_m_corr = m_corr - self.m_corr_fiducial_per_zbin[0]
            m_corr_per_zbin = np.zeros(self.nzbins)

            for zbin in range(0, self.nzbins):
                m_corr_per_zbin[zbin] = self.m_corr_fiducial_per_zbin[zbin] + delta_m_corr
        else:
            # if "m_corr" is not specified in input parameter script
            # we just apply the fiducial m-correction values
            # if these could not be loaded, this vector contains only zeros!
            m_corr_per_zbin = self.m_corr_fiducial_per_zbin

        index_corr = 0
        # a_noise_corr = np.zeros(self.nzcorrs)
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):
                # correlation = 'z{:}z{:}'.format(zbin1 + 1, zbin2 + 1)
                # calculate m-correction vector here:
                # this loop goes over bands per z-corr; m-correction is the
                # same for all bands in one tomographic bin!!!
                val_m_corr_ee = (1. + m_corr_per_zbin[zbin1]) * (
                    1. + m_corr_per_zbin[zbin2]) * np.ones(len(self.settings.bands_EE_to_use))
                val_m_corr_bb = (1. + m_corr_per_zbin[zbin1]) * (
                    1. + m_corr_per_zbin[zbin2]) * np.ones(len(self.settings.bands_BB_to_use))

                if index_corr == 0:
                    m_corr_ee = val_m_corr_ee
                    m_corr_bb = val_m_corr_bb
                else:
                    m_corr_ee = np.concatenate((m_corr_ee, val_m_corr_ee))
                    m_corr_bb = np.concatenate((m_corr_bb, val_m_corr_bb))

                index_corr += 1

        m_corr = np.concatenate((m_corr_ee, m_corr_bb))
        # this is required for scaling of covariance matrix:
        m_corr_matrix = np.matrix(m_corr).T * np.matrix(m_corr)

        return m_corr, m_corr_matrix

    def load_gps(self, folder_name):

        '''
        inputs:
            param folder_name (str): folder consisting of the GPs

        outputs:
            none
        '''
        all_gps = []

        for i in range(self.n_band):
            with open(folder_name + 'gp_' + str(i) + '.pkl', 'rb') as gp_load:
                gp_dummy = dill.load(gp_load)
            all_gps.append(gp_dummy)

        self.gg_tot = np.array(all_gps)

    def random_sample(self, point):

        '''
        inputs:
            param point (ndarray) - the test point

        outputs:
            param samples (ndarray) - samples from the GPs
        '''

        band_samples = []

        for i in range(self.n_band):
            band_samples.append(self.gg_tot[i].sampleBandPower(
                point, mean=False, nsamples=self.n_realisation))

        band_samples = np.array(band_samples).T

        band_samples_trans = np.zeros_like(band_samples)
        for i in range(self.n_realisation):
            band_samples_trans[i] = self.log_to_exp(band_samples[i])

        return band_samples_trans

    def mean_prediction(self, point):

        '''
        inputs:
            param point (ndarray) - the test point

        outputs:
            param mean (ndarray) - mean prediction from the GPs
        '''

        mean_tot = np.zeros(self.n_band)
        for i in range(self.n_band):
            mean_tot[i] = self.gg_tot[i].sampleBandPower(point, mean=True)

        c_total = self.log_to_exp(mean_tot)

        return c_total

    def log_to_exp(self, array_band):

        '''
        Function to do matrix logarithm transformation

        inputs:
            param array_band (ndarray) - a vector of bandpowers
        
        outputs:
            transformed bandpowers (ndarray)
        '''

        if np.isnan(array_band).any():
            return None

        else:

            n_corr = 6
            n_band = 4

            array_band = array_band.reshape(n_corr, n_band)

            tensor = np.zeros((n_band, self.nzbins, self.nzbins))
            expmatrix = np.zeros((n_corr, n_band))

            for i in range(n_band):
                tensor[i][0, 0] = array_band[:, i][0]
                tensor[i][1, 0] = array_band[:, i][1]
                tensor[i][1, 1] = array_band[:, i][2]
                tensor[i][2, 0] = array_band[:, i][3]
                tensor[i][2, 1] = array_band[:, i][4]
                tensor[i][2, 2] = array_band[:, i][5]

                for zbin1 in range(self.nzbins):
                    for zbin2 in range(self.nzbins):
                        tensor[i][zbin1, zbin2] = tensor[i][zbin2, zbin1]

                expmat = expm(tensor[i])

                index_corr = 0

                for zbin1 in range(self.nzbins):
                    for zbin2 in range(zbin1 + 1):
                        expmatrix[index_corr, i] = expmat[zbin1, zbin2]
                        index_corr += 1

            return expmatrix.flatten()

    def assign_parameters(self, parameters):

        self.cosmo_params = {'omega_cdm': float(parameters[0]),
                             'omega_b': float(parameters[1]),
                             'ln10^{10}A_s': float(parameters[2]),
                             'n_s': float(parameters[3]),
                             'h': float(parameters[4])
                             }

        param_name = 'm_corr'

        if self.settings.in_clsude_neutrino and (
                param_name in self.settings.use_nuisance):

            self.other_settings = {
                'N_ncdm': 1.0,
                'deg_ncdm': 3.0,
                'T_ncdm': 0.71611,
                'N_ur': 0.00641}
            self.neutrino_settings = {'m_ncdm': float(
                parameters[-1]) / self.other_settings['deg_ncdm']}

            self.systematics = {'A_bary': parameters[5],
                                'A_n1': parameters[6],
                                'A_n2': parameters[7],
                                'A_n3': parameters[8],
                                'A_IA': parameters[9],
                                'm': parameters[10]
                                }

        elif self.settings.in_clsude_neutrino and (param_name not in self.settings.use_nuisance):

            self.other_settings = {
                'N_ncdm': 1.0,
                'deg_ncdm': 3.0,
                'T_ncdm': 0.71611,
                'N_ur': 0.00641}
            self.neutrino_settings = {'m_ncdm': float(
                parameters[-1]) / self.other_settings['deg_ncdm']}

            self.systematics = {'A_bary': parameters[5],
                                'A_n1': parameters[6],
                                'A_n2': parameters[7],
                                'A_n3': parameters[8],
                                'A_IA': parameters[9],
                                }
        elif not self.settings.in_clsude_neutrino and (param_name in self.settings.use_nuisance):

            self.systematics = {'A_bary': parameters[5],
                                'A_n1': parameters[6],
                                'A_n2': parameters[7],
                                'A_n3': parameters[8],
                                'A_IA': parameters[9],
                                'm': parameters[10]
                                }

    def loglikelihood(self, parameters):

        self.assign_parameters(parameters)

        cosmo_ = self.cosmo_params.values()
        a_ia = self.systematics['A_IA']
        a_bary = self.systematics['A_bary']
        # careful here - we have to supply sum of neutrinos, that is,
        # parameters[-1], not 'm_ncdm'
        neut = parameters[-1]
        testpoint = np.concatenate([list(cosmo_), np.ones(
            1) * a_bary, np.ones(1) * neut, np.ones(1) * a_ia])

        index_ee = self.all_bands_ee_to_use == 1
        index_bb = self.all_bands_bb_to_use == 1

        if self.set_random:
            if self.n_realisation == 1:
                cl_ee_total = self.random_sample(testpoint).flatten()
            else:
                cl_ee_total = self.random_sample(testpoint)
        else:
            cl_ee_total = self.mean_prediction(testpoint)

        param_name = 'm_corr'

        if param_name in self.settings.use_nuisance:
            m_m, m_c = self.calc_m_correction()
            covariance = self.covariance / np.asarray(m_c)
            covariance = self.covariance[np.ix_(
                self.indices_for_bands_to_use, self.indices_for_bands_to_use)]

            band_powers = self.band_powers / np.asarray(m_m)
            band_powers = self.band_powers[self.indices_for_bands_to_use]
        else:
            band_powers = self.band_powers
            covariance = self.covariance

        cl_sys_bb, cl_sys_ee_noise, cl_sys_bb_noise = self.systematics_calc()

        theory_ee = cl_ee_total + cl_sys_ee_noise[index_ee]
        theory_bb = cl_sys_bb[index_bb] + cl_sys_bb_noise[index_bb]

        if (self.set_random and self.n_realisation > 1):
            theory_bb_nr = np.repeat(
                theory_bb.reshape(
                    1,
                    len(theory_bb)),
                self.n_realisation,
                axis=0)
            band_powers_theory = np.concatenate(
                (theory_ee, theory_bb_nr), axis=1)
            difference_vector = band_powers_theory - band_powers

        else:

            band_powers_theory = np.concatenate((theory_ee, theory_bb))
            difference_vector = band_powers_theory - band_powers

        if np.isinf(band_powers_theory).any() or np.isnan(
                band_powers_theory).any():
            return -1E32

        elif param_name in self.settings.use_nuisance:
            # use a Cholesky decomposition instead:
            chol_fact = cholesky(covariance, lower=True)

            if (self.set_random and self.n_realisation > 1):

                cinv = gpl.dpotrs(chol_fact, np.eye(chol_fact.shape[0]), lower=True)[0]
                cinv_diff = np.dot(cinv, difference_vector.T)
                chi2 = np.einsum('ij,ij->j', difference_vector.T, cinv_diff)

                return logsumexp(-0.5 * chi2) - np.log(self.n_realisation)

            else:
                yt = solve_triangular(chol_fact, difference_vector.T, lower=True)
                chi2 = yt.dot(yt)
                return -0.5 * chi2

    def priors(self):

        prior = OrderedDict()

        prior['omega_cdm'] = [0.010, 0.390, 'uniform']
        prior['omega_b'] = [0.019, 0.007, 'uniform']
        prior['ln10^{10}A_s'] = [1.700, 3.300, 'uniform']
        prior['n_s'] = [0.700, 0.600, 'uniform']
        prior['h'] = [0.640, 0.180, 'uniform']
        prior['A_bary'] = [0.000, 2.00, 'uniform']

        prior['A_n1'] = [-0.100, 0.200, 'uniform']
        prior['A_n2'] = [-0.100, 0.200, 'uniform']
        prior['A_n3'] = [-0.100, 0.200, 'uniform']
        prior['A_IA'] = [-6.00, 12.00, 'uniform']

        prior['m'] = [-0.033, 0.04, 'uniform']
        prior['m_ncdm'] = [0.06, 0.94, 'uniform']

        self.all_priors = genDist(prior)
        self.ndim = len(self.all_priors)

    def posterior(self, params):

        pri = [self.all_priors[i].pdf(params[i]) for i in range(len(params))]
        prodpri = np.prod(pri)

        if (prodpri == 0.0):
            return -1E32

        else:
            log_prod_prior = np.log(prodpri)
            loglike = self.loglikelihood(params)
            return loglike + log_prod_prior

    def emcee_sampler(self, guess, eps, nsamples=5, nwalkers=5):

        pos = [
            guess +
            eps *
            np.random.randn(
                self.ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.posterior)

        sampler.run_mcmc(pos, nsamples)

        return sampler

    def systematics_calc(self):

        a_noise = np.zeros(self.nzbins)
        add_noise_power = np.zeros(self.nzbins, dtype=bool)

        for zbin in range(self.nzbins):
            param_name = 'a_noise_z{:}'.format(zbin + 1)

            if param_name in self.settings.use_nuisance:
                a_noise[zbin] = self.systematics['A_n' + str(zbin + 1)]
                add_noise_power[zbin] = True

        # this was the fiducial approach for the first submission
        # the one above might be faster (and more consistent)
        if self.settings.correct_resetting_bias:
            amp_bb, exp_bb = np.random.multivariate_normal(
                self.params_resetting_bias, self.cov_resetting_bias)

        theory_bb = np.zeros((self.nzcorrs, self.band_offset_bb), 'float64')
        theory_noise_ee = np.zeros(
            (self.nzcorrs, self.band_offset_ee), 'float64')
        theory_noise_bb = np.zeros(
            (self.nzcorrs, self.band_offset_bb), 'float64')

        index_corr = 0
        # a_noise_corr = np.zeros(self.nzcorrs)
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):

                if zbin1 == zbin2:
                    a_noise_corr = a_noise[zbin1] * \
                        self.sigma_e[zbin1]**2 / self.n_eff[zbin1]
                else:
                    a_noise_corr = 0.

                # Cl_sample = Cl[:, zbin1, zbin2]
                # spline_Cl = itp.splrep(ells, Cl_sample)
                # d_l_EE = ell_norm * itp.splev(ells_sum, spline_Cl)
                # to do : 1e-9 can either become an adjustable constant or a parameter!
                # taking out ell_norm now (a constant times ell_norm is just
                # another noise-power component)
                if self.settings.correct_resetting_bias:
                    # to do : get ell_centers...

                    # x_bb = ell_center *
                    # (ell_center + 1.) / (2. * np.pi) *
                    # self.sigma_e[zbin1] *
                    # self.sigma_e[zbin2] / np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])

                    # try to pull the model through the BWM first, that's more
                    # consistent with the code and doesn't require
                    denom = np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])
                    numer = self.ell_norm * self.sigma_e[zbin1] * self.sigma_e[zbin2]
                    x_bb = numer / denom
                    d_l_bb = self.get_b_mode_model(x_bb, amp_bb, exp_bb)
                # else:
                #    d_l_bb = self.scale_B_modes # * ell_norm
                d_l_noise = self.ell_norm * a_noise_corr

                if self.settings.correct_resetting_bias:
                    theory_bb[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_bb, self.band_window_matrix, index_corr, band_type_ee=False)
                else:
                    theory_bb[index_corr, :] = 0.

                if add_noise_power.all():
                    theory_noise_ee[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_noise, self.band_window_matrix, index_corr, band_type_ee=True)
                    theory_noise_bb[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_noise, self.band_window_matrix, index_corr, band_type_ee=False)

                index_corr += 1

        return theory_bb.flatten(), theory_noise_ee.flatten(), theory_noise_bb.flatten()

    def get_b_mode_model(self, x, amp, exp):

        y = amp * x**exp

        return y

    # also for B-mode prediction:
    def get_theory(
            self,
            ells_sum,
            d_l,
            band_window_matrix,
            index_corr,
            band_type_ee=True):

        # these slice out the full EE --> EE and BB --> BB block of the full
        # BWM!
        sp_ee_x = (0, self.nzcorrs * self.band_offset_ee)
        sp_ee_y = (0, self.nzcorrs * len(self.ells_intp))
        sp_bb_x = (self.nzcorrs *
                   self.band_offset_ee, self.nzcorrs *
                   (self.band_offset_bb +
                    self.band_offset_ee))
        sp_bb_y = (self.nzcorrs *
                   len(self.ells_intp), 2 *
                   self.nzcorrs *
                   len(self.ells_intp))

        if band_type_ee:
            sp_x = sp_ee_x
            sp_y = sp_ee_y
            band_offset = self.band_offset_ee
        else:
            sp_x = sp_bb_x
            sp_y = sp_bb_y
            band_offset = self.band_offset_bb

        # print band_window_matrix
        # print band_window_matrix.shape

        bwm_sliced = band_window_matrix[sp_x[0]:sp_x[1], sp_y[0]:sp_y[1]]

        # print bwm
        # print bwm.shape

        # ell_norm = ells_sum * (ells_sum + 1) / (2. * np.pi)

        bands = range(index_corr * band_offset, (index_corr + 1) * band_offset)

        d_avg = np.zeros(len(bands))

        for index_band, alpha in enumerate(bands):
            # jump along tomographic auto-correlations only:
            index_ell_low = int(index_corr * len(self.ells_intp))
            index_ell_high = int((index_corr + 1) * len(self.ells_intp))
            spline_w_alpha_l = itp.splrep(
                self.ells_intp, bwm_sliced[alpha, index_ell_low:index_ell_high])
            # norm_val = np.sum(itp.splev(ells_sum, spline_w_alpha_l))
            # print 'Norm of W_al = {:.2e}'.format(norm_val)
            d_avg[index_band] = np.sum(
                itp.splev(ells_sum, spline_w_alpha_l) * d_l)

        return d_avg

    def post_process(self):
        '''
        Deletes the GPs when we store the MCMC samples
        '''
        del self.gg_tot


if __name__ == '__main__':


    FILENAME = 'mcmc_samples/bandpowers_emulator_mean_mcmc_samples'

    MCMC_SET = EMULATOR(
        directory=DIR,
        file_settings='scripts/settings',
        set_random=False,
        n_realisation=20)
    MCMC_SET.load_gps('gps/')
    MCMC_SET.priors()

    EPSILON = np.array([1E-3, 1E-4, 0.01, 0.01, 1E-3, 0.1, 1E-4, 1E-4, 1E-4, 0.1, 1E-4, 0.01])
    params = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 1.0078, 0.0289, 0.0133, -0.0087, -1.9163, 0.0, 0.5692])

    START = time.time()
    MCMC = MCMC_SET.emcee_sampler(
        params, EPSILON, nsamples=15000, nwalkers=24)
    print(MCMC.acceptance_fraction)
    print("--- %s seconds ---" % (time.time() - START))

    MCMC_SET.post_process()

    with open(FILENAME, 'wb') as g:
        dill.dump(MCMC, g)
