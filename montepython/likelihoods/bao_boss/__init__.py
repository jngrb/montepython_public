import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood


class bao_boss(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

	file_to_use = self.file
	for experiment in data.experiments:
	  if experiment == 'bao_boss_aniso':
	    print "WARNING: excluding isotropic CMASS measurement"
	    file_to_use = self.file_no_isotropic_cmass

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        for line in open(os.path.join(self.data_directory, file_to_use), 'r'):
            if (line.find('#') == -1):
                self.z = np.append(self.z, float(line.split()[0]))
                self.data = np.append(self.data, float(line.split()[1]))
                self.error = np.append(self.error, float(line.split()[2]))
                self.type = np.append(self.type, int(line.split()[3]))

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

            da = cosmo.angular_distance(self.z[i])
            dr = self.z[i] / cosmo.Hubble(self.z[i])
            dv = pow(da * da * (1 + self.z[i]) * (1 + self.z[i]) * dr, 1. / 3.)
            rs = cosmo.rs_drag()

            if (self.type[i] == 3):
                theo = dv / rs / self.rs_rescale

            elif (self.type[i] == 4):
                theo = dv

            elif (self.type[i] == 5):
                theo = da / rs

            elif (self.type[i] == 6):
                theo = 1. / cosmo.Hubble(self.z[i]) / rs

            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "BAO data type %s " % self.type[i] +
                    "in %d-th line not understood" % i)

            chi2 += ((theo - self.data[i]) / self.error[i]) ** 2

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl
