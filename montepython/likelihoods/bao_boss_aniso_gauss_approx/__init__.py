import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class bao_boss_aniso_gauss_approx(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

	# are there conflicting experiments?
	if 'bao_boss_aniso' in data.experiments:
	    raise io_mp.LikelihoodError('conflicting bao_boss_aniso measurments')

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.DA_in_Mpc = np.array([], 'float64')
        self.DA_error = np.array([], 'float64')
        self.H_in_km_per_s_per_Mpc = np.array([], 'float64')
        self.H_error = np.array([], 'float64')
        self.cross_corr = np.array([], 'float64')
        self.rs_fid_in_Mpc = np.array([], 'float64')

        # read redshifts and data points
        for line in open(os.path.join(self.data_directory, self.file), 'r'):
            if (line.find('#') == -1):
                self.z = np.append(self.z, float(line.split()[0]))
                self.DA_in_Mpc = np.append(self.DA_in_Mpc, float(line.split()[1]))
                self.DA_error = np.append(self.DA_error, float(line.split()[2]))
                self.H_in_km_per_s_per_Mpc = np.append(self.H_in_km_per_s_per_Mpc, float(line.split()[3]))
                self.H_error = np.append(self.H_error, float(line.split()[4]))
                self.cross_corr = np.append(self.cross_corr, float(line.split()[5]))
                self.rs_fid_in_Mpc = np.append(self.rs_fid_in_Mpc, float(line.split()[6]))

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.

        # for each point, compute angular distance da, radial distance dr,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # theoretical prediction and chi2 contribution
        for i in range(self.num_points):

            DA_at_z = cosmo.angular_distance(self.z[i])
            H_at_z = cosmo.Hubble(self.z[i]) * conts.c / 1000.0
            #dv = pow(da * da * (1 + self.z[i]) * (1 + self.z[i]) * dr, 1. / 3.)
            rs = cosmo.rs_drag() * self.rs_rescale

	    theo_DA = DA_at_z / rs * self.rs_fid_in_Mpc[i]
	    theo_H = H_at_z * rs / self.rs_fid_in_Mpc[i]

            chi2 += ((theo_DA - self.DA_in_Mpc[i]) / self.DA_error[i]) ** 2
            chi2 += ((theo_H - self.H_in_km_per_s_per_Mpc[i]) / self.H_error[i]) ** 2
            # account for cross correlation
            #chi2 += 2 * self.cross_corr[i] * (theo_DA - self.DA_in_Mpc[i]) * (theo_H - self.H_in_km_per_s_per_Mpc[i]) / self.DA_error[i] / self.H_error[i]

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl
