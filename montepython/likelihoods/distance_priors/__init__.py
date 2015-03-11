import os
import math
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.interpolate as interp
import scipy.constants as conts

class distance_priors(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        if type(distance_priors.centre1) is str:
            filepath = os.path.join(self.data_directory, self.centre1)
            self.centre1 = np.genfromtxt(filepath)
        if type(distance_priors.covmat1) is str:
            filepath = os.path.join(self.data_directory, self.covmat1)
            self.covmat1 = np.genfromtxt(filepath)
        self.inv_covmat1 = matrix(self.covmat1).I
        # FIXME: do the same for centre2,3 and covmat2,3

    def loglkl(self, cosmo, data):
        H0, ob, Om, ns, theta, da_rec, logAs = (
            data.mcmc_parameters[p]['current']*data.mcmc_parameters[p]['scale']
            for p in ['H0', 'omega_b', 'Omega_m', 'n_s', '100*theta_s', 'da_rec', 'ln10^{10}A_s'])
        R = math.sqrt(Om*H0**2) * da_rec * 1000.0 / conts.c
        print da_rec, R

        # The variables are, in order, omega_b*h**2, 100*theta_star, ns, R, ln(10**10 As)
        # where R = sqrt(Omega_m H_0**2)*d_A(z_star)*1000/c (if d_A is in Mpc and H0 in Mpc/(km/s))
        cur_data = [ob, theta, ns, R, logAs]

        # Modes
        lkl = 0
        #for centre, inv_covmat in zip([self.centre1, self.centre2, self.centre3],
                                      #[self.inv_covmat1, self.inv_covmat2, self.inv_covmat3]):
        for centre, inv_covmat in zip([self.centre1],
                                      [self.inv_covmat1]):
            diffvec = matrix([x-mu for x, mu in zip(cur_data, centre)])
            lkl += exp(-0.5 * (dot(diffvec, dot(inv_covmat, diffvec.T))))
        return log(lkl)
