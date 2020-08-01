from utils.genDistribution import genDist
import numpy as np
import scipy.stats as ss
from scipy.linalg import cholesky, solve_triangular
import scipy.interpolate as itp
from scipy.special import logsumexp
from GPy.util import linalg as gpl

import dill
from collections import OrderedDict
import emcee
import imp
import os
import time

# our script
import sys
sys.path.append('../../')

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, formatter={'float_kind': '{:0.2f}'.format})


class emulator:

    def __init__(self, Nband=24,file_settings='settings',set_random=False,n_realisation=10):

        self.Nband = Nband
        self.n_realisation = n_realisation
        # self.GPerror    = GPerror
        self.set_random = set_random

        self.file_settings = file_settings
        self.settings = imp.load_source(self.file_settings, self.file_settings)


        self.redshift_bins = []

        for index_zbin in range(len(self.settings.zbin_min)):
            redshift_bin = '{:.2f}z{:.2f}'.format(
                self.settings.zbin_min[index_zbin],
                self.settings.zbin_max[index_zbin])
            self.redshift_bins.append(redshift_bin)

        # number of z-bins
        self.nzbins = len(self.redshift_bins)
        # number of *unique* correlations between z-bins
        self.nzcorrs = int(self.nzbins * (self.nzbins + 1) / 2)

        self.all_bands_EE_to_use = []
        self.all_bands_BB_to_use = []

        # default, use all correlations:
        for i in range(self.nzcorrs):
            self.all_bands_EE_to_use += self.settings.bands_EE_to_use
            self.all_bands_BB_to_use += self.settings.bands_BB_to_use

        self.all_bands_EE_to_use = np.array(self.all_bands_EE_to_use)
        self.all_bands_BB_to_use = np.array(self.all_bands_BB_to_use)

        all_bands_to_use = np.concatenate((self.all_bands_EE_to_use, self.all_bands_BB_to_use))
        self.indices_for_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]
        # print(self.indices_for_bands_to_use)

        # this is also the number of points in the datavector
        ndata = len(self.indices_for_bands_to_use)

        if self.settings.correct_resetting_bias:
            fname = os.path.join(self.settings.data_directory, 'Resetting_bias/parameters_B_mode_model.dat')
            A_B_modes, exp_B_modes, err_A_B_modes, err_exp_B_modes = np.loadtxt(fname, unpack=True)
            self.params_resetting_bias = np.array([A_B_modes, exp_B_modes])

            fname = os.path.join(self.settings.data_directory, 'Resetting_bias/covariance_B_mode_model.dat')
            self.cov_resetting_bias = np.loadtxt(fname)

        # try to load fiducial m-corrections from file
        # (currently these are global values over full field, hence no looping over fields required for that!)
        # TODO: Make output dependent on field, not necessary for current KiDS approach though!
        try:
            fname = os.path.join(self.settings.data_directory, '{:}zbins/m_correction_avg.txt'.format(self.nzbins))

            if self.nzbins == 1:
                self.m_corr_fiducial_per_zbin = np.asarray([np.loadtxt(fname, usecols=[1])])
            else:
                self.m_corr_fiducial_per_zbin = np.loadtxt(fname, usecols=[1])
        except BaseException:
            self.m_corr_fiducial_per_zbin = np.zeros(self.nzbins)
            print('Could not load m-correction values from {}\n'.format(fname))
            print('Setting them to zero instead.')

        try:
            fname = os.path.join(self.settings.data_directory,
                                 '{:}zbins/sigma_int_n_eff_{:}zbins.dat'.format(self.nzbins, self.nzbins))
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

        collect_bp_EE_in_zbins = []
        collect_bp_BB_in_zbins = []
        # collect BP per zbin and combine into one array
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):
                # zbin2 first in fname!
                i, j, k = self.nzbins, zbin1 + 1, zbin2 + 1
                fname_EE = os.path.join(self.settings.data_directory,
                                        '{:}zbins/band_powers_EE_z{:}xz{:}.dat'.format(i, j, k))
                fname_BB = os.path.join(self.settings.data_directory,
                                        '{:}zbins/band_powers_BB_z{:}xz{:}.dat'.format(i, j, k))
                extracted_band_powers_EE = np.loadtxt(fname_EE)
                extracted_band_powers_BB = np.loadtxt(fname_BB)
                collect_bp_EE_in_zbins.append(extracted_band_powers_EE)
                collect_bp_BB_in_zbins.append(extracted_band_powers_BB)

        self.band_powers = np.concatenate(
            (np.asarray(collect_bp_EE_in_zbins).flatten(),
             np.asarray(collect_bp_BB_in_zbins).flatten()))

        fname = os.path.join(self.settings.data_directory, '{:}zbins/covariance_all_z_EE_BB.dat'.format(self.nzbins))
        self.covariance = np.loadtxt(fname)

        fname = os.path.join(self.settings.data_directory,
                             '{:}zbins/band_window_matrix_nell100.dat'.format(self.nzbins))
        self.band_window_matrix = np.loadtxt(fname)
        # ells_intp and also band_offset are consistent between different patches!

        fname = os.path.join(self.settings.data_directory,
                             '{:}zbins/multipole_nodes_for_band_window_functions_nell100.dat'.format(self.nzbins))
        self.ells_intp = np.loadtxt(fname)

        self.band_offset_EE = len(extracted_band_powers_EE)
        self.band_offset_BB = len(extracted_band_powers_BB)

        # Read fiducial dn_dz from window files:
        # TODO: the hardcoded z_min and z_max correspond to the lower and upper
        # endpoints of the shifted left-border histogram!
        z_samples = []
        hist_samples = []
        for zbin in range(self.nzbins):
            redshift_bin = self.redshift_bins[zbin]
            window_file_path = os.path.join(self.settings.data_directory,
                                            '{:}/n_z_avg_{:}.hist'.format(self.settings.photoz_method, redshift_bin))
            zptemp, hist_pz = np.loadtxt(window_file_path, usecols=[0, 1], unpack=True)
            shift_to_midpoint = np.diff(zptemp)[0] / 2.
            # we add a zero as first element because we want to integrate down to z = 0!
            z_samples += [np.concatenate((np.zeros(1), zptemp + shift_to_midpoint))]
            hist_samples += [np.concatenate((np.zeros(1), hist_pz))]

        z_samples = np.asarray(z_samples)
        hist_samples = np.asarray(hist_samples)

        # prevent undersampling of histograms!
        if self.settings.nzmax < len(zptemp):
            cmnt = "You're trying to integrate at lower resolution than supplied " + \
                "by the n(z) histograms. \n Increase nzmax! Aborting now..."
            print(cmnt)
        # if that's the case, we want to integrate at histogram resolution and need to account for
        # the extra zero entry added
        elif self.settings.nzmax == len(zptemp):
            self.settings.nzmax = z_samples.shape[1]
            # requires that z-spacing is always the same for all bins...
            self.redshifts = z_samples[0, :]
            print('Integrations performed at resolution of histogram!')
            # if we interpolate anyway at arbitrary resolution the extra 0 doesn't matter
        else:
            self.settings.nzmax += 1
            self.redshifts = np.linspace(z_samples.min(), z_samples.max(), self.settings.nzmax)
            print('Integration performed at set nzmax resolution!')

        self.pz = np.zeros((self.settings.nzmax, self.nzbins))
        self.pz_norm = np.zeros(self.nzbins, 'float64')

        for zbin in range(self.nzbins):
            # we assume that the histograms loaded are given as left-border histograms
            # and that the z-spacing is the same for each histogram
            spline_pz = itp.splrep(z_samples[zbin, :], hist_samples[zbin, :])

            # z_mod = self.z_p
            mask_min = self.redshifts >= z_samples[zbin, :].min()
            mask_max = self.redshifts <= z_samples[zbin, :].max()
            mask = mask_min & mask_max
            # points outside the z-range of the histograms are set to 0!
            self.pz[mask, zbin] = itp.splev(self.redshifts[mask], spline_pz)
            # Normalize selection functions
            dz = self.redshifts[1:] - self.redshifts[:-1]
            self.pz_norm[zbin] = np.sum(0.5 * (self.pz[1:, zbin] + self.pz[:-1, zbin]) * dz)

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

        fname = os.path.join(self.settings.data_directory, 'number_datapoints.txt')
        np.savetxt(fname, [ndata], header='number of datapoints in masked datavector')

        # 1) determine l-range for taking the sum, #l = l_high-l_min at least!!!:
        # this is the correct calculation!
        # for real data, I should start sum from physical scales, i.e., currently l>= 80!
        # TODO: Set this automatically!!!
        # not automatically yet, but controllable via "myCFHTLenS_tomography.data"!!!
        # these are integer l-values over which we will take the sum used in the convolution with the band window matrix
        self.ells_min = self.ells_intp[0]
        self.ells_max = self.ells_intp[-1]
        self.nells = int(self.ells_max - self.ells_min + 1)
        self.ells_sum = np.linspace(self.ells_min, self.ells_max, self.nells)

        # these are the l-nodes for the derivation of the theoretical Cl:
        self.ells = np.logspace(np.log10(self.ells_min), np.log10(self.ells_max), self.settings.nellsmax)
        self.ell_norm = self.ells_sum * (self.ells_sum + 1) / (2. * np.pi)

        self.bands_EE_selected = np.tile(self.settings.bands_EE_to_use, self.nzcorrs)
        self.bands_BB_selected = np.tile(self.settings.bands_BB_to_use, self.nzcorrs)

        # Account for m-correction if it is fixed to fiducial values

        if 'm_corr' not in self.settings.use_nuisance:
            m_m, m_c = self.calc_m_correction()
            self.covariance = self.covariance / np.asarray(m_c)
            self.covariance = self.covariance[np.ix_(self.indices_for_bands_to_use, self.indices_for_bands_to_use)]
            self.band_powers = self.band_powers / np.asarray(m_m)
            self.band_powers = self.band_powers[self.indices_for_bands_to_use]
            self.covInverse = np.linalg.inv(self.covariance)
            self.L = cholesky(self.covariance, lower=True)

    def calc_m_correction(self):

        param_name = 'm_corr'
        if param_name in self.settings.use_nuisance:
            m_corr = self.systematics['m']
            delta_m_corr = m_corr - self.m_corr_fiducial_per_zbin[0]
            m_corr_per_zbin = np.zeros(self.nzbins)

            for zbin in range(0, self.nzbins):
                m_corr_per_zbin[zbin] = self.m_corr_fiducial_per_zbin[zbin] + delta_m_corr
        else:
            # if "m_corr" is not specified in input parameter script we just apply the fiducial m-correction values
            # if these could not be loaded, this vector contains only zeros!
            m_corr_per_zbin = self.m_corr_fiducial_per_zbin

        index_corr = 0
        # A_noise_corr = np.zeros(self.nzcorrs)
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):
                # correlation = 'z{:}z{:}'.format(zbin1 + 1, zbin2 + 1)
                # calculate m-correction vector here:
                # this loop goes over bands per z-corr; m-correction is the same for all bands in one tomographic bin!!!
                pref = (1. + m_corr_per_zbin[zbin1]) * (1. + m_corr_per_zbin[zbin2])
                val_m_corr_EE = pref * np.ones(len(self.settings.bands_EE_to_use))
                val_m_corr_BB = pref * np.ones(len(self.settings.bands_BB_to_use))

                if index_corr == 0:
                    m_corr_EE = val_m_corr_EE
                    m_corr_BB = val_m_corr_BB
                else:
                    m_corr_EE = np.concatenate((m_corr_EE, val_m_corr_EE))
                    m_corr_BB = np.concatenate((m_corr_BB, val_m_corr_BB))

                index_corr += 1

        m_corr = np.concatenate((m_corr_EE, m_corr_BB))
        # this is required for scaling of covariance matrix:
        m_corr_matrix = np.matrix(m_corr).T * np.matrix(m_corr)

        return m_corr, m_corr_matrix

    def assignParameters(self, parameters):

        self.cosmoParams = {'omega_cdm': float(parameters[0]),
                            'omega_b': float(parameters[1]),
                            'ln10^{10}A_s': float(parameters[2]),
                            'n_s': float(parameters[3]),
                            'h': float(parameters[4])
                            }

        param_name = 'm_corr'

        if self.settings.include_neutrino and (param_name in self.settings.use_nuisance):

            self.other_settings = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}
            self.neutrino_settings = {'m_ncdm': float(parameters[-1]) / self.other_settings['deg_ncdm']}

            self.systematics = {'A_bary': parameters[5],
                                'A_n1': parameters[6],
                                'A_n2': parameters[7],
                                'A_n3': parameters[8],
                                'A_IA': parameters[9],
                                'm': parameters[10]
                                }

        elif self.settings.include_neutrino and (param_name not in self.settings.use_nuisance):

            self.other_settings = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}
            self.neutrino_settings = {'m_ncdm': float(parameters[-1]) / self.other_settings['deg_ncdm']}

            self.systematics = {'A_bary': parameters[5],
                                'A_n1': parameters[6],
                                'A_n2': parameters[7],
                                'A_n3': parameters[8],
                                'A_IA': parameters[9],
                                }
        elif not self.settings.include_neutrino and (param_name in self.settings.use_nuisance):

            self.systematics = {'A_bary': parameters[5],
                                'A_n1': parameters[6],
                                'A_n2': parameters[7],
                                'A_n3': parameters[8],
                                'A_IA': parameters[9],
                                'm': parameters[10]
                                }

    def systematicsCalc(self):

        A_noise = np.zeros(self.nzbins)
        add_noise_power = np.zeros(self.nzbins, dtype=bool)

        for zbin in range(self.nzbins):
            param_name = 'A_noise_z{:}'.format(zbin + 1)

            if param_name in self.settings.use_nuisance:
                A_noise[zbin] = self.systematics['A_n' + str(zbin + 1)]
                add_noise_power[zbin] = True

        # this was the fiducial approach for the first submission
        # the one above might be faster (and more consistent)
        if self.settings.correct_resetting_bias:
            # A_B_modes = np.random.normal(self.best_fit_A_B_modes, self.best_fit_err_A_B_modes)
            # exp_B_modes = np.random.normal(self.best_fit_exp_B_modes, self.best_fit_err_exp_B_modes)
            amp_BB, exp_BB = np.random.multivariate_normal(self.params_resetting_bias, self.cov_resetting_bias)
            # print 'resetting bias'
            # print self.params_resetting_bias, self.cov_resetting_bias
            # print amp_BB, exp_BB

        # param_name = 'm_corr'
        # if param_name in self.settings.use_nuisance:
        # 	m_corr          = self.systematics['m']
        # 	delta_m_corr    = m_corr - self.m_corr_fiducial_per_zbin[0]
        # 	m_corr_per_zbin = np.zeros(self.nzbins)

        # 	for zbin in range(0, self.nzbins):
        # 		m_corr_per_zbin[zbin] = self.m_corr_fiducial_per_zbin[zbin] + delta_m_corr
        # else:
        #     # if "m_corr" is not specified in input parameter script we just apply the fiducial m-correction values
        #     # if these could not be loaded, this vector contains only zeros!
        #     m_corr_per_zbin = self.m_corr_fiducial_per_zbin

        theory_BB = np.zeros((self.nzcorrs, self.band_offset_BB), 'float64')
        theory_noise_EE = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')
        theory_noise_BB = np.zeros((self.nzcorrs, self.band_offset_BB), 'float64')

        index_corr = 0

        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):
                if zbin1 == zbin2:
                    A_noise_corr = A_noise[zbin1] * self.sigma_e[zbin1]**2 / self.n_eff[zbin1]
                else:
                    A_noise_corr = 0.
                # Cl_sample = Cl[:, zbin1, zbin2]
                # spline_Cl = itp.splrep(ells, Cl_sample)
                # D_l_EE = ell_norm * itp.splev(ells_sum, spline_Cl)
                # TODO: 1e-9 can either become an adjustable constant or a parameter!
                # taking out ell_norm now (a constant times ell_norm is just another noise-power component)
                if self.settings.correct_resetting_bias:
                    # TODO: get ell_centers...

                    # x_BB = ell_center *
                    # (ell_center + 1.) / (2. * np.pi) *
                    # self.sigma_e[zbin1] *
                    # self.sigma_e[zbin2] / np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])

                    # try to pull the model through the BWM first, that's more
                    # consistent with the code and doesn't require
                    denom = np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])
                    numer = self.ell_norm * self.sigma_e[zbin1] * self.sigma_e[zbin2]
                    x_BB = numer / denom
                    D_l_BB = self.get_B_mode_model(x_BB, amp_BB, exp_BB)
                # else:
                #    D_l_BB = self.scale_B_modes # * ell_norm
                D_l_noise = self.ell_norm * A_noise_corr

                if self.settings.correct_resetting_bias:
                    theory_BB[index_corr, :] = self.get_theory(
                        self.ells_sum, D_l_BB, self.band_window_matrix, index_corr, band_type_is_EE=False)
                else:
                    theory_BB[index_corr, :] = 0.

                if add_noise_power.all():
                    theory_noise_EE[index_corr, :] = self.get_theory(
                        self.ells_sum, D_l_noise, self.band_window_matrix, index_corr, band_type_is_EE=True)
                    theory_noise_BB[index_corr, :] = self.get_theory(
                        self.ells_sum, D_l_noise, self.band_window_matrix, index_corr, band_type_is_EE=False)

                # if index_corr == 0:
                # 	m_corr_EE = val_m_corr_EE
                # 	m_corr_BB = val_m_corr_BB
                # else:
                # 	m_corr_EE = np.concatenate((m_corr_EE, val_m_corr_EE))
                # 	m_corr_BB = np.concatenate((m_corr_BB, val_m_corr_BB))

                index_corr += 1

        # take care of m-correction:
        # m_corr = np.concatenate((m_corr_EE, m_corr_BB))
        # # this is required for scaling of covariance matrix:
        # m_corr_matrix = np.matrix(m_corr).T * np.matrix(m_corr)

        # theory_BB = theory_BB.flatten() + theory_noise_BB.flatten()
        # theory_EE = theory_EE.flatten() + theory_noise_EE.flatten()
        # band_powers_theory = np.concatenate((theory_EE, theory_BB))

        # we want elementwise division!!!
        # covariance = self.covariance / np.asarray(m_corr_matrix)

        # some numpy-magic for slicing:
        # cov_sliced = covariance[np.ix_(self.indices_for_bands_to_use, self.indices_for_bands_to_use)]

        return theory_BB.flatten(), theory_noise_EE.flatten(), theory_noise_BB.flatten()

    def get_B_mode_model(self, x, amp, exp):

        y = amp * x**exp

        return y

    # also for B-mode prediction:
    def get_theory(self, ells_sum, D_l, band_window_matrix, index_corr, band_type_is_EE=True):

        # these slice out the full EE --> EE and BB --> BB block of the full BWM!
        sp_EE_x = (0, self.nzcorrs * self.band_offset_EE)
        sp_EE_y = (0, self.nzcorrs * len(self.ells_intp))
        sp_BB_x = (self.nzcorrs * self.band_offset_EE,
                   self.nzcorrs * (self.band_offset_BB + self.band_offset_EE))
        sp_BB_y = (self.nzcorrs * len(self.ells_intp), 2 * self.nzcorrs * len(self.ells_intp))

        if band_type_is_EE:
            sp_x = sp_EE_x
            sp_y = sp_EE_y
            band_offset = self.band_offset_EE
        else:
            sp_x = sp_BB_x
            sp_y = sp_BB_y
            band_offset = self.band_offset_BB

        bwm_sliced = band_window_matrix[sp_x[0]:sp_x[1], sp_y[0]:sp_y[1]]

        bands = range(index_corr * band_offset, (index_corr + 1) * band_offset)

        D_avg = np.zeros(len(bands))

        for index_band, alpha in enumerate(bands):
            # jump along tomographic auto-correlations only:
            index_ell_low = int(index_corr * len(self.ells_intp))
            index_ell_high = int((index_corr + 1) * len(self.ells_intp))
            spline_w_alpha_l = itp.splrep(self.ells_intp, bwm_sliced[alpha, index_ell_low:index_ell_high])
            # norm_val = np.sum(itp.splev(ells_sum, spline_w_alpha_l))
            # print 'Norm of W_al = {:.2e}'.format(norm_val)
            D_avg[index_band] = np.sum(itp.splev(ells_sum, spline_w_alpha_l) * D_l)

        return D_avg

    def loadGPs(self, folderName):
        allGPs = []

        for i in range(11):
            with open(folderName + 'gp_' + str(i) + '.pkl', 'rb') as g:
                gp_dummy = dill.load(g)
            allGPs.append(gp_dummy)

        self.gg_tot_moped = np.array(allGPs)

    def priors_moped(self):

        prior = OrderedDict()
        prior['omega_cdm'] = [0.010, 0.390, 'uniform']
        prior['omega_b'] = [0.019, 0.007, 'uniform']
        prior['ln10^{10}A_s'] = [1.700, 3.300, 'uniform']
        prior['n_s'] = [0.700, 0.600, 'uniform']
        prior['h'] = [0.640, 0.180, 'uniform']
        prior['A_bary'] = [0.00, 2.0, 'uniform']

        prior['A_n1'] = [-0.100, 0.200, 'uniform']
        prior['A_n2'] = [-0.100, 0.200, 'uniform']
        prior['A_n3'] = [-0.100, 0.200, 'uniform']
        prior['A_IA'] = [-6.00, 12.00, 'uniform']

        prior['m_ncdm'] = [0.06, 0.94, 'uniform']

        self.allPrior = genDist(prior)
        self.ndim = len(self.allPrior)

    def randomSample_x_moped(self, point):

        band_samples = []

        for i in range(len(self.gg_tot_moped)):
            band_samples.append(self.gg_tot_moped[i].sampleBandPower(point, mean=False, nsamples=self.n_realisation))

        band_samples = np.array(band_samples).T

        return band_samples

    def meanPredictionMoped(self, point):

        mean_tot = np.zeros(11)
        for i in range(11):
            mean_tot[i] = self.gg_tot_moped[i].sampleBandPower(point, mean=True)

        return mean_tot

    def theory_moped(self, parameters):

        self.assignParameters(parameters)

        cosmo_ = self.cosmoParams.values()
        a_ia = self.systematics['A_IA']
        a_bary = self.systematics['A_bary']

        # careful here - we have to supply sum of neutrinos, that is, parameters[-1], not 'm_ncdm'
        neut = parameters[-1]
        testpoint = np.concatenate([list(cosmo_), np.ones(1) * a_bary, np.ones(1) * neut, np.ones(1) * a_ia])

        if self.set_random:
            if self.n_realisation == 1:
                moped_ee_total = self.randomSample_x_moped(testpoint).flatten()
            else:
                moped_ee_total = self.randomSample_x_moped(testpoint)
        else:
            moped_ee_total = self.meanPredictionMoped(testpoint)

        cl_sys_bb, cl_sys_ee_noise, cl_sys_bb_noise = self.systematicsCalc()

        index_ee = self.all_bands_EE_to_use == 1
        index_bb = self.all_bands_BB_to_use == 1

        band_sys = np.concatenate((cl_sys_ee_noise[self.bands_EE_selected == 1],
                                   cl_sys_bb_noise[self.bands_BB_selected == 1]))
        toAdd = np.dot(self.b[:, 24:], cl_sys_bb[self.bands_BB_selected == 1]) + np.dot(self.b, band_sys)

        return moped_ee_total + toAdd

    def loadMopedVectors(self, fileName='1.txt'):
        self.b = np.loadtxt('moped_b_' + fileName)
        self.y_s = np.loadtxt('moped_y_' + fileName)

    def logLike_moped(self, parameters):

        theory = self.theory_moped(parameters)

        difference_vector = self.y_s - theory

        if (self.set_random and self.n_realisation > 1):
            chi2 = np.sum(difference_vector**2, axis=1)
            return logsumexp(-0.5 * chi2) - np.log(self.n_realisation)

        else:
            chi2 = np.sum(difference_vector**2)
            return -0.5 * chi2

    def posterior_moped(self, params):

        pri = [self.allPrior[i].pdf(params[i]) for i in range(len(params))]
        prodpri = np.prod(pri)

        if (prodpri == 0.0):
            logprodPri = -1E32
        else:
            logprodPri = np.log(prodpri)

        loglike = self.logLike_moped(params)

        return loglike + logprodPri

    def deleteGPs(self):
        del self.gg_tot_moped

    def emceeSampler_moped(self, guess, eps, nSamples=5, nwalkers=5):

        pos = [guess + eps * np.random.randn(self.ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.posterior_moped)
        sampler.run_mcmc(pos, nSamples)

        return sampler


if __name__ == '__main__':

    filename = 'mcmc_samples/samples_emulator_moped_1000'

    emulator_moped = emulator(file_settings='settingsMoped',set_random=False, n_realisation=20)
    emulator_moped.loadGPs('gps/')
    emulator_moped.priors_moped()

    params = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 1.0078, 0.0289, 0.0133, -0.0087, -1.9163, 0.5692])
    emulator_moped.loadMopedVectors(fileName='3.txt')
    # emulator_moped.theory_moped(params)
    print(emulator_moped.logLike_moped(params))

    # testTheory = emulator_moped.theory_moped()

    # print(emulator_moped.b.shape)

    # start_time = time.time()
    # print(emulator_moped.logLike_moped(params))
    # print("--- %s seconds ---" % (time.time() - start_time))

    # print(emulator_moped.posterior_moped(params))

    # eps =  np.array([1E-3, 1E-4, 0.01, 0.01, 1E-3, 0.1, 1E-4, 1E-4, 1E-4, 0.1, 0.01])

    # start_time = time.time()
    # wholeSampler = emulator_moped.emceeSampler_moped(params, eps, nSamples = 15000, nwalkers = 22)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # emulator_moped.postProcessMoped()

    # with open(workDirectory+filename, 'wb') as g:
    # 	dill.dump(wholeSampler, g)
