import sys
import gc
import emcee
from utils.genDistribution import genDist
import scipy.optimize as op
import dill
from collections import OrderedDict
from classy import Class
import imp
from scipy.linalg import cholesky, solve_triangular
import scipy.interpolate as itp
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, formatter={'float_kind': '{:0.2f}'.format})


class kids450:

    def __init__(self, file_settings='settings'):

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

        all_bands_EE_to_use = []
        all_bands_BB_to_use = []

        # default, use all correlations:
        for i in range(self.nzcorrs):
            all_bands_EE_to_use += self.settings.bands_EE_to_use
            all_bands_BB_to_use += self.settings.bands_BB_to_use

        all_bands_to_use = np.concatenate((all_bands_EE_to_use, all_bands_BB_to_use))
        self.indices_for_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]

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
            cmnt = "You're trying to integrate at lower resolution than " + \
                "supplied by the n(z) histograms. \n Increase nzmax! Aborting now..."
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
        #  not automatically yet, but controllable via "myCFHTLenS_tomography.data"!!!
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

    def fitEE(self):

        cosmo = Class()
        cosmo.set(self.cosmoParams)
        cosmo.set(self.class_argumets)

        if self.settings.include_neutrino:
            cosmo.set(self.other_settings)
            cosmo.set(self.neutrino_settings)

        cosmo.compute()

        self.Omega_m = cosmo.Omega_m()
        self.small_h = cosmo.h()

        self.rho_crit = self.get_critical_density()
        # derive the linear growth factor D(z)
        linear_growth_rate = np.zeros_like(self.redshifts)
        # print self.redshifts
        for index_z, z in enumerate(self.redshifts):
            try:
                # for CLASS ver >= 2.6:
                linear_growth_rate[index_z] = cosmo.scale_independent_growth_factor(z)
            except BaseException:
                # my own function from private CLASS modification:
                linear_growth_rate[index_z] = cosmo.growth_factor_at_z(z)
        # normalize to unity at z=0:
        try:
            # for CLASS ver >= 2.6:
            linear_growth_rate /= cosmo.scale_independent_growth_factor(0.)
        except BaseException:
            # my own function from private CLASS modification:
            linear_growth_rate /= cosmo.growth_factor_at_z(0.)

        # get distances from cosmo-module:
        r, dzdr = cosmo.z_of_r(self.redshifts)

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        # this doesn't really have to go into the loop over fields!
        pk = np.zeros((self.settings.nellsmax, self.settings.nzmax), 'float64')
        k_max_in_inv_Mpc = self.settings.k_max_h_by_Mpc * self.small_h

        for index_ells in range(self.settings.nellsmax):
            for index_z in range(1, self.settings.nzmax):
                # standard Limber approximation:
                # k = ells[index_ells] / r[index_z]
                # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                k_in_inv_Mpc = (self.ells[index_ells] + 0.5) / r[index_z]
                if k_in_inv_Mpc > k_max_in_inv_Mpc:
                    pk_dm = 0.
                else:
                    pk_dm = cosmo.pk(k_in_inv_Mpc, self.redshifts[index_z])
                # pk[index_ells,index_z] = cosmo.pk(ells[index_ells]/r[index_z], self.redshifts[index_z])
                if self.settings.baryon_feedback:
                    pk[index_ells,
                       index_z] = pk_dm * self.baryon_feedback_bias_sqr(k_in_inv_Mpc / self.small_h,
                                                                        self.redshifts[index_z],
                                                                        A_bary=self.systematics['A_bary'])
                else:
                    pk[index_ells, index_z] = pk_dm

    # for KiDS-450 constant biases in photo-z are not sufficient:
        if self.settings.bootstrap_photoz_errors:
            # draw a random bootstrap n(z); borders are inclusive!
            random_index_bootstrap = np.random.randint(
                int(self.settings.index_bootstrap_low), int(self.settings.index_bootstrap_high) + 1)
            # print 'Bootstrap index:', random_index_bootstrap
            pz = np.zeros((self.settings.nzmax, self.nzbins), 'float64')
            pz_norm = np.zeros(self.nzbins, 'float64')

            for zbin in range(self.nzbins):

                redshift_bin = self.redshift_bins[zbin]
                # ATTENTION: hard-coded subfolder!
                # index can be recycled since bootstraps for tomographic bins are independent!
                fname = os.path.join(self.settings.data_directory,
                                     '{:}/bootstraps/{:}/n_z_avg_bootstrap{:}.hist'.format(self.settings.photoz_method,
                                                                                           redshift_bin,
                                                                                           random_index_bootstrap))
                z_hist, n_z_hist = np.loadtxt(fname, unpack=True)

                shift_to_midpoint = np.diff(z_hist)[0] / 2.
                spline_pz = itp.splrep(z_hist + shift_to_midpoint, n_z_hist)
                mask_min = self.redshifts >= z_hist.min() + shift_to_midpoint
                mask_max = self.redshifts <= z_hist.max() + shift_to_midpoint
                mask = mask_min & mask_max
                # points outside the z-range of the histograms are set to 0!
                pz[mask, zbin] = itp.splev(self.redshifts[mask], spline_pz)

                dz = self.redshifts[1:] - self.redshifts[:-1]
                pz_norm[zbin] = np.sum(0.5 * (pz[1:, zbin] + pz[:-1, zbin]) * dz)

            pr = pz * (dzdr[:, np.newaxis] / pz_norm)

        else:
            pr = self.pz * (dzdr[:, np.newaxis] / self.pz_norm)

        g = np.zeros((self.settings.nzmax, self.nzbins), 'float64')

        for zbin in range(self.nzbins):
            # assumes that z[0] = 0
            for nr in range(1, self.settings.nzmax - 1):
                # for nr in range(self.nzmax - 1):
                fun = pr[nr:, zbin] * (r[nr:] - r[nr]) / r[nr:]
                g[nr, zbin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (r[nr + 1:] - r[nr:-1]))
                g[nr, zbin] *= 2. * r[nr] * (1. + self.redshifts[nr])

        # Start loop over l for computation of C_l^shear
        Cl_GG_integrand = np.zeros((self.settings.nzmax, self.nzbins, self.nzbins), 'float64')
        Cl_GG = np.zeros((self.settings.nellsmax, self.nzbins, self.nzbins), 'float64')

        Cl_II_integrand = np.zeros_like(Cl_GG_integrand)
        Cl_II = np.zeros_like(Cl_GG)

        Cl_GI_integrand = np.zeros_like(Cl_GG_integrand)
        Cl_GI = np.zeros_like(Cl_GG)

        dr = r[1:] - r[:-1]
        for index_ell in range(self.settings.nellsmax):

            # find Cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
            for zbin1 in range(self.nzbins):
                for zbin2 in range(zbin1 + 1):  # self.nzbins):
                    Cl_GG_integrand[1:, zbin1, zbin2] = g[1:, zbin1] * g[1:, zbin2] / r[1:]**2 * pk[index_ell, 1:]

                    factor_IA = self.get_factor_IA(self.redshifts[1:],
                                                   linear_growth_rate[1:],
                                                   self.systematics['A_IA'])  # / self.dzdr[1:]
                    # print F_of_x
                    # print self.eta_r[1:, zbin1].shape
                    Cl_II_integrand[1:, zbin1, zbin2] = pr[1:, zbin1] * \
                        pr[1:, zbin2] * factor_IA**2 / r[1:]**2 * pk[index_ell, 1:]
                    pref = g[1:, zbin1] * pr[1:, zbin2] + g[1:, zbin2] * pr[1:, zbin1]
                    Cl_GI_integrand[1:, zbin1, zbin2] = pref * factor_IA / r[1:]**2 * pk[index_ell, 1:]

            # Integrate over r to get C_l^shear_ij = P_ij(l)
            # C_l^shear_ii = 9/4 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r) g_j(r) /r**2) P(k=l/r,z(r))
            for zbin1 in range(self.nzbins):
                for zbin2 in range(zbin1 + 1):  # self.nzbins):
                    Cl_GG[index_ell, zbin1, zbin2] = np.sum(
                        0.5 * (Cl_GG_integrand[1:, zbin1, zbin2] + Cl_GG_integrand[:-1, zbin1, zbin2]) * dr)
                    # here we divide by 16, because we get a 2^2 from g(z)!
                    Cl_GG[index_ell, zbin1, zbin2] *= 9. / 16. * self.Omega_m**2  # in units of Mpc**4
                    Cl_GG[index_ell, zbin1, zbin2] *= (self.small_h / 2997.9)**4  # dimensionless

                    Cl_II[index_ell, zbin1, zbin2] = np.sum(
                        0.5 * (Cl_II_integrand[1:, zbin1, zbin2] + Cl_II_integrand[:-1, zbin1, zbin2]) * dr)

                    Cl_GI[index_ell, zbin1, zbin2] = np.sum(
                        0.5 * (Cl_GI_integrand[1:, zbin1, zbin2] + Cl_GI_integrand[:-1, zbin1, zbin2]) * dr)
                    # here we divide by 4, because we get a 2 from g(r)!
                    Cl_GI[index_ell, zbin1, zbin2] *= 3. / 4. * self.Omega_m
                    Cl_GI[index_ell, zbin1, zbin2] *= (self.small_h / 2997.9)**2

        # ordering of redshift bins is correct in definition of theory below!
        theory_EE_GG = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')
        theory_EE_II = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')
        theory_EE_GI = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')

        index_corr = 0
        # A_noise_corr = np.zeros(self.nzcorrs)
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):
                # correlation = 'z{:}z{:}'.format(zbin1 + 1, zbin2 + 1)

                Cl_sample_GG = Cl_GG[:, zbin1, zbin2]
                spline_Cl_GG = itp.splrep(self.ells, Cl_sample_GG)
                D_l_EE_GG = self.ell_norm * itp.splev(self.ells_sum, spline_Cl_GG)

                theory_EE_GG[index_corr, :] = self.get_theory(
                    self.ells_sum, D_l_EE_GG, self.band_window_matrix, index_corr, band_type_is_EE=True)

                Cl_sample_GI = Cl_GI[:, zbin1, zbin2]
                spline_Cl_GI = itp.splrep(self.ells, Cl_sample_GI)
                D_l_EE_GI = self.ell_norm * itp.splev(self.ells_sum, spline_Cl_GI)
                theory_EE_GI[index_corr, :] = self.get_theory(
                    self.ells_sum, D_l_EE_GI, self.band_window_matrix, index_corr, band_type_is_EE=True)

                Cl_sample_II = Cl_II[:, zbin1, zbin2]
                spline_Cl_II = itp.splrep(self.ells, Cl_sample_II)
                D_l_EE_II = self.ell_norm * itp.splev(self.ells_sum, spline_Cl_II)
                theory_EE_II[index_corr, :] = self.get_theory(
                    self.ells_sum, D_l_EE_II, self.band_window_matrix, index_corr, band_type_is_EE=True)

                index_corr += 1

        cosmo.struct_cleanup()
        cosmo.empty()

        del cosmo
        gc.collect()

        return theory_EE_GG.flatten(), theory_EE_GI.flatten(), theory_EE_II.flatten()

    def priors(self):

        # 00 - omega_cdm_h^2
        # 01 - omega_b_h^2
        # 02 - ln(10^10 A_s)
        # 03 - n_s
        # 04 - h
        # 05 - A_bary
        # 06 - A_n1
        # 07 - A_n2
        # 08 - A_n3
        # 09 - A_IA
        # 10 - sum(neutrinos)

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

        prior['m_ncdm'] = [0.06, 0.94, 'uniform']

        self.allPrior = genDist(prior)
        self.ndim = len(self.allPrior)

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

        theory_BB = np.zeros((self.nzcorrs, self.band_offset_BB), 'float64')
        theory_noise_EE = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')
        theory_noise_BB = np.zeros((self.nzcorrs, self.band_offset_BB), 'float64')

        index_corr = 0
        # A_noise_corr = np.zeros(self.nzcorrs)
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):  # self.nzbins):
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
                    x_BB = self.ell_norm * self.sigma_e[zbin1] * self.sigma_e[zbin2] / \
                        np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])
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

                index_corr += 1

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

        # print band_window_matrix
        # print band_window_matrix.shape

        bwm_sliced = band_window_matrix[sp_x[0]:sp_x[1], sp_y[0]:sp_y[1]]

        # print bwm
        # print bwm.shape

        # ell_norm = ells_sum * (ells_sum + 1) / (2. * np.pi)

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

    def baryon_feedback_bias_sqr(self, k, z, A_bary=1.):
        """
        Fitting formula for baryon feedback following equation 10
        and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)
        """
        baryon_model = self.settings.baryon_model

        # k is expected in h/Mpc and is divided in log by this unit...
        x = np.log10(k)

        a = 1. / (1. + z)
        a_sqr = a * a

        constant = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
                            'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
                            'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
                    'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
                            'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
                            'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
                    'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                              'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                              'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}

        A_z = constant[baryon_model]['A2'] * a_sqr + constant[baryon_model]['A1'] * a + constant[baryon_model]['A0']
        B_z = constant[baryon_model]['B2'] * a_sqr + constant[baryon_model]['B1'] * a + constant[baryon_model]['B0']
        C_z = constant[baryon_model]['C2'] * a_sqr + constant[baryon_model]['C1'] * a + constant[baryon_model]['C0']
        D_z = constant[baryon_model]['D2'] * a_sqr + constant[baryon_model]['D1'] * a + constant[baryon_model]['D0']
        E_z = constant[baryon_model]['E2'] * a_sqr + constant[baryon_model]['E1'] * a + constant[baryon_model]['E0']

        # only for debugging; tested and works!
        # print 'AGN: A2=-0.11900, B2= 0.1300, C2= 0.6000, D2= 0.002110, E2=-2.0600'
        # print baryon_model+': A2={:.5f}, B2={:.5f}, C2={:.5f}, D2={:.5f},
        # E2={:.5f}'.format(constant[baryon_model]['A2'],
        # constant[baryon_model]['B2'],
        # constant[baryon_model]['C2'],constant[baryon_model]['D2'],
        # constant[baryon_model]['E2'])

        # original formula:
        # bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
        # original formula with a free amplitude A_bary:
        bias_sqr = 1. - A_bary * (A_z * np.exp((B_z * x - C_z)**3) - D_z * x * np.exp(E_z * x))

        return bias_sqr

    def get_factor_IA(self, z, linear_growth_rate, amplitude, exponent=0.0):

        const = 5e-14 / self.small_h**2  # in Mpc^3 / M_sol

        # arbitrary convention
        z0 = 0.3
        # print utils.growth_factor(z, self.Omega_m)
        # print self.rho_crit
        factor = -1. * amplitude * const * self.rho_crit * self.Omega_m / \
            linear_growth_rate * ((1. + z) / (1. + z0))**exponent

        return factor

    def get_critical_density(self):
        """
        The critical density of the Universe at redshift 0.

        Returns
        -------
        rho_crit in solar masses per cubic Megaparsec.

        """

        # yay, constants...
        Mpc_cm = 3.08568025e24  # cm
        M_sun_g = 1.98892e33  # g
        G_const_Mpc_Msun_s = M_sun_g * (6.673e-8) / Mpc_cm**3.
        H100_s = 100. / (Mpc_cm * 1.0e-5)  # s^-1

        rho_crit_0 = 3. * (self.small_h * H100_s)**2. / (8. * np.pi * G_const_Mpc_Msun_s)

        return rho_crit_0

    def theory(self, parameters):

        self.assignParameters(parameters)

        gg, gi, ii = self.fitEE()
        cl_ee_total = gg + gi + ii

        cl_sys_bb, cl_sys_ee_noise, cl_sys_bb_noise = self.systematicsCalc()

        theory_EE = cl_ee_total + cl_sys_ee_noise
        theory_BB = cl_sys_bb + cl_sys_bb_noise
        band_powers_theory = np.concatenate((theory_EE, theory_BB))

        return band_powers_theory[self.indices_for_bands_to_use]

    def mopedCoeff(self, parameters):

        self.assignParameters(parameters)

        gg, gi, ii = self.fitEE()
        cl_ee_total = gg + gi + ii

        # cl_sys_bb, cl_sys_ee_noise, cl_sys_bb_noise = self.systematicsCalc()

        # theory_EE = cl_ee_total + cl_sys_ee_noise
        # theory_BB = cl_sys_bb   + cl_sys_bb_noise
        # band_powers_theory = np.concatenate((theory_EE, theory_BB))

        # band_sys = np.concatenate((cl_sys_ee_noise[self.bands_EE_selected == 1],
        #                           cl_sys_bb_noise[self.bands_BB_selected == 1]))
        # toAdd = np.dot(self.b[:,24:], cl_sys_bb[self.bands_BB_selected==1]) + np.dot(self.b, band_sys)
        return np.dot(self.b[:, 0:24], cl_ee_total[self.bands_EE_selected == 1])

    def gradient(self, parameters, deltaTheta=1E-6, option='central'):

        grad = np.zeros((self.ndim, len(self.band_powers)))
        Cinv_Grad_mu = np.zeros((self.ndim, len(self.band_powers)))
        grad_Cinv_grad = np.zeros(self.ndim)
        bs = np.zeros((self.ndim, len(self.band_powers)))

        if (option == 'central'):

            for i in range(self.ndim):

                parameters_plus = np.copy(parameters)
                parameters_minus = np.copy(parameters)
                parameters_plus[i] = parameters_plus[i] + deltaTheta
                parameters_minus[i] = parameters_minus[i] - deltaTheta

                cl_total_plus = self.theory(parameters_plus)
                cl_total_minus = self.theory(parameters_minus)
                grad[i] = (cl_total_plus - cl_total_minus) / (2.0 * deltaTheta)

                Cinv_Grad_mu[i] = np.dot(self.covInverse, grad[i])
                grad_Cinv_grad[i] = np.dot(grad[i], Cinv_Grad_mu[i])

        else:

            cl_total, b_data, cov_ = self.theory(parameters)

            for i in range(self.ndim):

                parameters_plus = np.copy(parameters)
                parameters_plus[i] = parameters_plus[i] + deltaTheta

                cl_total_plus = self.theory(parameters_plus)

                grad[i] = (cl_total_plus - cl_total) / deltaTheta
                Cinv_Grad_mu[i] = np.dot(self.covInverse, grad[i])
                grad_Cinv_grad[i] = np.dot(grad[i], Cinv_Grad_mu[i])

        for i in range(self.ndim):

            if (i == 0):
                bs[i] = Cinv_Grad_mu[i] / np.sqrt(grad_Cinv_grad[i])

            else:

                dummy_numerator = np.zeros((len(self.band_powers), int(i)))
                dummy_denominator = np.zeros(int(i))

                for j in range(i):
                    dummy_numerator[:, j] = np.dot(grad[i], bs[j]) * bs[j]
                    dummy_denominator[j] = np.dot(grad[i], bs[j])**2

                bs[i] = (Cinv_Grad_mu[i] - np.sum(dummy_numerator, axis=1)) / \
                    np.sqrt(grad_Cinv_grad[i] - np.sum(dummy_denominator))

        y_alphas = np.dot(bs, self.band_powers)

        for i in range(self.ndim):
            for j in range(i + 1):
                if i == j:
                    print('Dot product between {0} and {1} is :{2:.10f}'.format(
                        i, j, np.dot(bs[i], np.dot(self.covariance, bs[j]))))

        self.b = bs
        self.y_s = y_alphas

        return bs, y_alphas

    def saveMopedVectors(self, fileName='1.txt'):
        np.savetxt('moped_b_' + fileName, self.b)
        np.savetxt('moped_y_' + fileName, self.y_s)
        return None

    def loadMopedVectors(self, fileName='1.txt'):
        self.b = np.loadtxt('moped_b_' + fileName)
        self.y_s = np.loadtxt('moped_y_' + fileName)

    def theory_moped(self, parameters):

        c_total = self.theory(parameters)

        return np.dot(self.b, c_total)

    def logLike_moped(self, parameters):

        self.assignParameters(parameters)

        pri = [self.allPrior[i].pdf(parameters[i]) for i in range(len(parameters))]
        prodpri = np.prod(pri)

        if (prodpri == 0.0):

            chi2 = 1E32

        else:

            theory = self.theory_moped(parameters)
            chi2 = np.sum((self.y_s - theory)**2)

            if (np.isnan(chi2) or np.isinf(chi2)):
                chi2 = 1E32

        return -0.5 * chi2

    def posterior_moped(self, params):

        pri = [self.allPrior[i].pdf(params[i]) for i in range(len(params))]
        prodpri = np.prod(pri)

        if (prodpri == 0.0):
            logprodPri = -1E32
        else:
            logprodPri = np.log(prodpri)

        loglike = self.logLike_moped(params)

        logPost = loglike + logprodPri

        return logPost

    def emceeSampler_moped(self, guess, eps, nSamples=5, nwalkers=5):

        pos = [guess + eps * np.random.randn(self.ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.posterior_moped)
        sampler.run_mcmc(pos, nSamples)

        return sampler
