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

        if type(self.centre1) is str:
            filepath = os.path.join(self.data_directory, self.centre1)
            self.centre1 = np.genfromtxt(filepath)
        if type(self.covmat1) is str:
            filepath = os.path.join(self.data_directory, self.covmat1)
            self.covmat1 = np.genfromtxt(filepath)
        self.inv_covmat1 = np.matrix(self.covmat1).I
        # FIXME: do the same for centre2,3 and covmat2,3
        print self.inv_covmat1 * np.matrix(self.covmat1)
        print np.matrix(self.covmat1) * self.inv_covmat1

        # !DEBUG OUT!
        #print self.centre1

    def loglkl(self, cosmo, data):
        H0, ob, Om, ns, theta, ra_rec, logAs = (
            data.mcmc_parameters[p]['current']*data.mcmc_parameters[p]['scale']
            for p in ['H0', 'omega_b', 'Omega_m', 'n_s', '100*theta_s', 'ra_rec', 'ln10^{10}A_s'])
        # NOTE: ra_rec is in Mpc
        R = math.sqrt(Om*H0**2) * ra_rec * 1e3 / conts.c # 1/(m/s) to 1/(km/s) => factor of 1000

        # The variables are, in order, Omega_b*h**2, 100*theta_star, ns, R, ln(10**10 As)
        # where R = sqrt(Omega_m H_0**2)*d_A(z_star)*1e6/c (if d_A is in Gpc and H0 in (km/s)/Mpc)
        cur_data = np.array([ob, theta, ns, R, logAs])

        # !DEBUG OUT!
        fid_data = np.array([0.0222253, 1.040346, 0.96445, 1.748745, 3.094129])
        print fid_data
        #print self.centre1
        #print cur_data

        # Modes
        loglkl = 0
        #for centre, inv_covmat in zip([self.centre1, self.centre2, self.centre3],
                                      #[self.inv_covmat1, self.inv_covmat2, self.inv_covmat3]):
        for centre, inv_covmat in zip([self.centre1],
                                      [self.inv_covmat1]):
            #diffvec = np.matrix([x-mu for x, mu in zip(cur_data, centre)])
            diffvec = np.matrix(cur_data - centre)

            # !DEBUG OUT!
            #print diffvec
            diffvec_fid = np.matrix(fid_data - centre)
            print diffvec_fid

            loglkl += -0.5 * (np.dot(diffvec, np.dot(inv_covmat, diffvec.T)))[0,0]

            # !DEBUG OUT!
            print -0.5 * (np.dot(diffvec_fid, np.dot(inv_covmat, diffvec_fid.T)))[0,0]

        return loglkl
