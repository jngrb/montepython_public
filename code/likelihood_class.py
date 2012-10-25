import os,sys
try:
  from collections import OrderedDict as od
except:
  from ordereddict import OrderedDict as od
import numpy as np
import math

import data
import io

# Definition of classes for the likelihoods to inherit, for every different
# types of expected likelihoods. Included so far, models for a Clik, simple
# prior, and newdat likelihoods.

# General class that will only define the store_lkl function, common for every
# single likelihood. It will copy the content of self.path, copied from
# initialization
class likelihood():

  def __init__(self,path,data,command_line,log_flag,default):

    self.name = self.__class__.__name__
    self.folder = os.path.abspath(data.path['MontePython'])+'/../likelihoods/'+self.name+'/'
    if (not default or not log_flag):
      path = command_line.folder+'log.param'
    self.read_from_file(path,data)

    # Check if the nuisance parameters are defined
    try :
      for nuisance in self.use_nuisance:
	if nuisance not in data.get_mcmc_parameters(['nuisance']):
	  print nuisance+' must be defined, either fixed or varying, for {0} likelihood'.format(self.name)
	  exit()
    except AttributeError:
      pass

    # Append to the log.param the value used (WARNING: so far no comparison is
    # done to ensure that the experiments share the same parameters)
    if log_flag: 
      io.log_likelihood_parameters(self,command_line)

  # This is a placeholder, to remind that, for a brand new likelihood, you need
  # to define its computation.
  def loglkl(self,_cosmo,data):
    raise NotImplementedError('Must implement method loglkl() in your likelihood')

  def read_from_file(self,path,data):
    self.path = path
    self.dictionary={}
    if os.path.isfile(path):
      data_file = open(path,'r')
      for line in data_file:
	if line.find('#')==-1:
	  if line.find(self.name+'.')!=-1:
	    exec(line.replace(self.name+'.','self.'))
	    # This part serves only to compare
	    key = line.split('=')[0].strip(' ').strip('\t').strip('\n').split('.')[1]
	    value = line.split('=')[-1].strip(' ').strip('\t').strip('\n')
	    self.dictionary[key] = value
      data_file.seek(0)
      data_file.close()
    try:
      if (self.data_directory[-1]!='/'):
	self.data_directory[-1]+='/'
    except:
      pass

  def get_cl(self,_cosmo):

    # get C_l^XX from the cosmological code
    cl = _cosmo.lensed_cl()

    # convert dimensionless C_l's to C_l in muK**2
    T = _cosmo._T_cmb()
    for key in cl.iterkeys():
      cl[key] *= (T*1.e6)**2

    return cl

  # Ensure that certain arguments of the cosmological code are defined to the
  # needed value.  WARNING: so far, there is no way to enforce a parameter
  # where smaller is better. A bigger value will always override any smaller
  # one (cl_max, etc...) TODO
  def need_cosmo_arguments(self,data,dictionary):
    array_flag = False
    for key,value in dictionary.iteritems():
      try :
	data.cosmo_arguments[key]
	try:
	  float(data.cosmo_arguments[key])
	  num_flag = True
	except ValueError: num_flag = False
	except TypeError:  
	  num_flag = True
	  array_flag = True
	
      except KeyError:
	try:
	  float(value)
	  num_flag = True
	  data.cosmo_arguments[key] = 0
	except ValueError:
	  num_flag = False
	  data.cosmo_arguments[key] = ''
	except TypeError:
	  num_flag = True
	  array_flag = True
      if num_flag is False:
	if data.cosmo_arguments[key].find(value)==-1:
	  data.cosmo_arguments[key] += ' '+value+' '
      else:
	if array_flag is False:
	  if float(data.cosmo_arguments[key])<value:
	    data.cosmo_arguments[key] = value
	else:
	  data.cosmo_arguments[key] = '%.2g' % value[0]
	  for i in range(1,len(value)):
	    data.cosmo_arguments[key] += ',%.2g' % (value[i])



  def read_contamination_spectra(self,data):

    for nuisance in self.use_nuisance:
      # read spectrum contamination (so far, assumes only temperature contamination; will be trivial to generalize to polarization when such templates will become relevant)
      exec "self.%s_contamination=np.zeros(self.l_max+1,'float64')" % nuisance
      try:  
        exec "File = open(self.data_directory+self.%s_file,'r')" % nuisance
        for line in File:
          l=int(float(line.split()[0]))
          if ((l >=2) and (l <= self.l_max)):
            exec "self.%s_contamination[l]=float(line.split()[1])/(l*(l+1.)/2./math.pi)" % nuisance

      except:
	print 'Warning: you did not pass a file name containing a contamination spectrum regulated by the nuisance parameter '+nuisance

      # read renormalization factor
      # if it is not there, assume it is one, i.e. do not renormalize  
      try:
        exec "self.%s_contamination *= float(self.%s_scale)" % (nuisance,nuisance)
      except:
        pass

      # read central value of nuisance parameter
      # if it is not there, assume one by default
      try:
        exec "self.%s_prior_center" % nuisance
      except:
        exec "self.%s_prior_center=1." % nuisance

      # read variance of nuisance parameter
      # if it are not there, assume flat prior (encoded through variance=0)
      try:
        exec "self.%s_prior_variance" % nuisance
      except:
        exec "self.%s_prior_variance=0." % nuisance

  def add_contamination_spectra(self,cl,data):

    # Recover the current value of the nuisance parameter. 
    for nuisance in self.use_nuisance:
      nuisance_value = float(data.mcmc_parameters[nuisance]['current']*data.mcmc_parameters[nuisance]['initial'][4])

      # add contamination spectra multiplied by nuisance parameters
      for l in range(2,self.l_max):
        exec "cl['tt'][l] += nuisance_value*self.%s_contamination[l]" % nuisance

    return cl

  def add_nuisance_prior(self,lkl,data):

    # Recover the current value of the nuisance parameter. 
    for nuisance in self.use_nuisance:
      nuisance_value = float(data.mcmc_parameters[nuisance]['current']*data.mcmc_parameters[nuisance]['initial'][4])

      # add prior on nuisance parameters
      exec "if (self.%s_prior_variance>0): lkl += -0.5*((nuisance_value-self.%s_prior_center)/self.%s_prior_variance)**2" % (nuisance,nuisance,nuisance)

    return lkl


###################################
#
# END OF GENERIC LIKELIHOOD CLASS
#
###################################



###################################
# PRIOR TYPE LIKELIHOOD 
# --> H0,...
###################################
class likelihood_prior(likelihood):

  def loglkl(self):
    raise NotImplementedError('Must implement method loglkl() in your likelihood')


###################################
# NEWDAT TYPE LIKELIHOOD
# --> spt,boomerang,etc.
###################################
class likelihood_newdat(likelihood):

  def __init__(self,path,data,command_line,log_flag,default):

    likelihood.__init__(self,path,data,command_line,log_flag,default)

    self.need_cosmo_arguments(data,{'lensing':'yes', 'output':'tCl lCl pCl'})
    
    if not default:
      return
      
    # open .newdat file
    newdatfile=open(self.data_directory+self.file,'r')

    # find beginning of window functions file names
    window_name=newdatfile.readline().strip('\n').replace(' ','')
    
    # initialize list of fist and last band for each type
    band_num = np.zeros(6,'int')
    band_min = np.zeros(6,'int')
    band_max = np.zeros(6,'int')

    # read number of bands for each of the six types TT, EE, BB, EB, TE, TB
    line = newdatfile.readline()
    for i in range(6):
      band_num[i] = int(line.split()[i])

    # read string equal to 'BAND_SELECTION' or not
    line = str(newdatfile.readline()).strip('\n').replace(' ','')

    # if yes, read 6 lines containing 'min, max'
    if (line=='BAND_SELECTION'):
      for i in range(6):
        line = newdatfile.readline()
        band_min[i] = int(line.split()[0])   
        band_max[i] = int(line.split()[1])

    # if no, set min to 1 and max to band_num (=use all bands)   
    else:
      band_min=[1 for i in range(6)]
      band_max=band_num

    # read line defining calibration uncertainty
    # contains: flag (=0 or 1), calib, calib_uncertainty
    line = newdatfile.readline()
    calib=float(line.split()[1])
    if (int(line.split()[0])==0):
      self.calib_uncertainty=0
    else:
      self.calib_uncertainty=float(line.split()[2])
 
    # read line defining beam uncertainty
    # contains: flag (=0, 1 or 2), beam_width, beam_sigma
    line = newdatfile.readline()
    beam_type = int(line.split()[0])
    if (beam_type > 0):
      self.has_beam_uncertainty = True
    else:
      self.has_beam_uncertainty = False
    beam_width = float(line.split()[1])
    beam_sigma = float(line.split()[2])
    
    # read flag (= 0, 1 or 2) for lognormal distributions and xfactors
    line = newdatfile.readline()
    likelihood_type = int(line.split()[0])
    if (likelihood_type > 0):
      self.has_xfactors = True
    else:
      self.has_xfactors = False

    # declare array of quantitites describing each point of measurement
    # size yet unknown, it will be found later and stored as self.num_points
    self.obs=np.array([],'float64')
    self.var=np.array([],'float64')
    self.beam_error=np.array([],'float64')
    self.has_xfactor=np.array([],'bool')
    self.xfactor=np.array([],'float64')

    # temporary array to know which bands are actually used
    used_index=np.array([],'int')

    index=-1

    # scan the lines describing each point of measurement
    for cltype in range(6):
      if (int(band_num[cltype]) != 0):
        # read name (but do not use it)
        newdatfile.readline()
        for band in range(int(band_num[cltype])):
          # read one line corresponding to one measurement
          line = newdatfile.readline()
          index += 1

          # if we wish to actually use this measurement
          if ((band >= band_min[cltype]-1) and (band <= band_max[cltype]-1)):

            used_index=np.append(used_index,index)

            self.obs=np.append(self.obs,float(line.split()[1])*calib**2)

            self.var=np.append(self.var,(0.5*(float(line.split()[2])+float(line.split()[3]))*calib**2)**2)

            self.xfactor=np.append(self.xfactor,float(line.split()[4])*calib**2)

            if ((likelihood_type == 0) or ((likelihood_type == 2) and (int(line.split()[7])==0))):
              self.has_xfactor=np.append(self.has_xfactor,[False])
            if ((likelihood_type == 1) or ((likelihood_type == 2) and (int(line.split()[7])==1))):
              self.has_xfactor=np.append(self.has_xfactor,[True])

            if (beam_type == 0):
              self.beam_error=np.append(self.beam_error,0.)
            if (beam_type == 1):
              l_mid=float(line.split()[5])+0.5*(float(line.split()[5])+float(line.split()[6]))
              self.beam_error=np.append(self.beam_error,abs(math.exp(-l_mid*(l_mid+1)*1.526e-8*2.*beam_sigma*beam_width)-1.))
            if (beam_type == 2):
              if (likelihood_type == 2):
                self.beam_error=np.append(self.beam_error,float(line.split()[8]))
              else:
                self.beam_error=np.append(self.beam_error,float(line.split()[7]))

        # now, skip and unused part of the file (with sub-correlation matrices)
        for band in range(int(band_num[cltype])):      
          newdatfile.readline()

    # number of points that we will actually use
    self.num_points=np.shape(self.obs)[0]

    # total number of points, including unused ones
    full_num_points=index+1

    # read full correlation matrix
    full_covmat=np.zeros((full_num_points,full_num_points),'float64')
    for point in range(full_num_points):
      full_covmat[point]=newdatfile.readline().split()

    # extract smaller correlation matrix for points actually used
    covmat=np.zeros((self.num_points,self.num_points),'float64')
    for point in range(self.num_points):
      covmat[point]=full_covmat[used_index[point],used_index]

    # recalibrate this correlation matrix
    covmat *= calib**4

    # redefine the correlation matrix, the observed points and their variance in case of lognormal likelihood
    if (self.has_xfactors):

      for i in range(self.num_points):

        for j in range(self.num_points):
          if (self.has_xfactor[i]):
            covmat[i,j] /= (self.obs[i]+self.xfactor[i])
          if (self.has_xfactor[j]):
            covmat[i,j] /= (self.obs[j]+self.xfactor[j])
            
      for i in range(self.num_points):
        if (self.has_xfactor[i]):
          self.var[i] /= (self.obs[i]+self.xfactor[i])**2
          self.obs[i] = math.log(self.obs[i]+self.xfactor[i])

    # invert correlation matrix
    self.inv_covmat=np.linalg.inv(covmat)

    # read window function files a first time, only for finding the smallest and largest l's for each point
    self.win_min=np.zeros(self.num_points,'int')
    self.win_max=np.zeros(self.num_points,'int')
    for point in range(self.num_points):
      for line in open(self.data_directory+'windows/'+window_name+str(used_index[point]+1),'r'):
        if any([float(line.split()[i]) != 0. for i in range(1,len(line.split()))]):
          if (self.win_min[point]==0):
            self.win_min[point]=int(line.split()[0])
          self.win_max[point]=int(line.split()[0])

    # infer from format of window function files whether we will use polarisation spectra or not 
    num_col=len(line.split())
    if (num_col == 2):
      self.has_pol=False
    else:
      if (num_col == 5):
        self.has_pol=True
      else:
        print 'window function files are understood of they contain 2 columns (l TT)'
        print 'or 5 columns (l TT TE EE BB)'
        print 'in this case the number of columns is',num_col
        exit()

    # define array of window functions
    self.window=np.zeros((self.num_points,max(self.win_max)+1,num_col-1),'float64')

    # go again through window function file, this time reading window functions; 
    # that are distributed as: l TT (TE EE BB)
    # where the last columns contaim W_l/l, not W_l
    # we mutiply by l in order to store the actual W_l
    for point in range(self.num_points):
      for line in open(self.data_directory+'windows/'+window_name+str(used_index[point]+1),'r'):
        l=int(line.split()[0])
        if (((self.has_pol==False) and (len(line.split()) !=2)) or ((self.has_pol==True) and (len(line.split()) !=5))):
           print 'for given experiment, all window functions should have the same number of columns, 2 or 5. This is not the case here.'
           exit()
        if ((l>=self.win_min[point]) and (l<=self.win_max[point])):
          self.window[point,l,:]=[float(line.split()[i]) for i in range(1,len(line.split()))]
          self.window[point,l,:]*=l

    # eventually, initialise quantitites used in the marginalization over nuisance parameters
    if ((self.has_xfactors == True) and ((self.calib_uncertainty > 1.e-4) or (self.has_beam_uncertainty == True))):
      self.halfsteps=5
      self.margeweights = np.zeros(2*self.halfsteps+1,'float64') 
      for i in range(-self.halfsteps,self.halfsteps+1):
        self.margeweights[i+self.halfsteps]=np.exp(-(float(i)*3./float(self.halfsteps))**2/2)
      self.margenorm=sum(self.margeweights)

    # store maximum value of l needed by window functions
    self.l_max=max(self.win_max)  

    # impose that the cosmological code computes Cl's up to maximum l needed by
    # the window function
    self.need_cosmo_arguments(data,{'l_max_scalars':self.l_max})

    # deal with nuisance parameters
    try:
      self.use_nuisance
    except:
      self.use_nuisance = []
    self.read_contamination_spectra(data)

    # end of initialisation

  def loglkl(self,_cosmo,data):

    # get Cl's from the cosmological code
    cl = self.get_cl(_cosmo)

    # add contamination spectra multiplied by nuisance parameters
    cl = self.add_contamination_spectra(cl,data)

    # get likelihood
    lkl = self.compute_lkl(cl,_cosmo,data)

    # add prior on nuisance parameters
    lkl = self.add_nuisance_prior(lkl,data)

    return lkl



  def compute_lkl(self,cl,_cosmo,data):

    # checks that Cl's have been computed up to high enough l given window function range. Normally this has been imposed before, so this test could even be supressed.
    if (np.shape(cl['tt'])[0]-1 < self.l_max):
      print '%s computed Cls till l=' % data.cosmological_module_name,np.shape(cl['tt'])[0]-1,'while window functions need',self.l_max
      exit()

    # compute theoretical bandpowers, store them in theo[points]
    theo=np.zeros(self.num_points,'float64')

    for point in range(self.num_points):

      # find bandpowers B_l by convolving C_l's with [(l+1/2)/2pi W_l]
      for l in range(self.win_min[point],self.win_max[point]):

        theo[point] += cl['tt'][l]*self.window[point,l,0]*(l+0.5)/2./math.pi

        if (self.has_pol):
          theo[point] += (cl['te'][l]*self.window[point,l,1] + cl['ee'][l]*self.window[point,l,2] + cl['bb'][l]*self.window[point,l,3])*(l+0.5)/2./math.pi

    # allocate array for differencve between observed and theoretical bandpowers
    difference=np.zeros(self.num_points,'float64')

    # depending on the presence of lognormal likelihood, calibration uncertainty and beam uncertainity, use several methods for marginalising over nuisance parameters:

    # first method: numerical integration over calibration uncertainty:
    if (self.has_xfactors and ((self.calib_uncertainty > 1.e-4) or self.has_beam_uncertainty)):
    
      chisq_tmp=np.zeros(2*self.halfsteps+1,'float64')
      chisqcalib=np.zeros(2*self.halfsteps+1,'float64')
      beam_error=np.zeros(self.num_points,'float64')

      # loop over various beam errors
      for ibeam in range(2*self.halfsteps+1):

        # beam error
        for point in range(self.num_points):
          if (self.has_beam_uncertainty):
            beam_error[point]=1.+self.beam_error[point]*(ibeam-self.halfsteps)*3/float(self.halfsteps)
          else:
            beam_error[point]=1.

        # loop over various calibraion errors
        for icalib in range(2*self.halfsteps+1):

          # calibration error
          calib_error=1+self.calib_uncertainty*(icalib-self.halfsteps)*3/float(self.halfsteps)
         
          # compute difference between observed and theoretical points, after correcting the later for errors
          for point in range(self.num_points):

            # for lognormal likelihood, use log(B_l+X_l)
            if (self.has_xfactor[point]):
              difference[point]=self.obs[point]-math.log(theo[point]*beam_error[point]*calib_error+self.xfactor[point])
            # otherwise use B_l
            else:
              difference[point]=self.obs[point]-theo[point]*beam_error[point]*calib_error

              # find chisq with those corrections
         #chisq_tmp[icalib]=np.dot(np.transpose(difference),np.dot(self.inv_covmat,difference))
          chisq_tmp[icalib]=np.dot(difference,np.dot(self.inv_covmat,difference))

        minchisq=min(chisq_tmp)

       # find chisq marginalized over calibration uncertainty (if any)
        tot=0
        for icalib in range(2*self.halfsteps+1):
          tot += self.margeweights[icalib]*math.exp(max(-30.,-(chisq_tmp[icalib]-minchisq)/2.))

        chisqcalib[ibeam]=-2*math.log(tot/self.margenorm)+minchisq

     # find chisq marginalized over beam uncertainty (if any)
      if (self.has_beam_uncertainty):

        minchisq=min(chisqcalib)

        tot=0
        for ibeam in range(2*self.halfsteps+1):
          tot += self.margeweights[ibeam]*math.exp(max(-30.,-(chisqcalib[ibeam]-minchisq)/2.))

        chisq=-2*math.log(tot/self.margenorm)+minchisq

      else:
        chisq=chisqcalib[0]

   # second method: marginalize over nuisance parameters (if any) analytically
    else:

     # for lognormal likelihood, theo[point] should contain log(B_l+X_l)
      if (self.has_xfactors): 
        for point in range(self.num_points):
          if (self.has_xfactor[point]):
            theo[point]=math.log(theo[point]+self.xfactor[point])
 
     # find vector of difference between observed and theoretical bandpowers
      difference=self.obs-theo  

     # find chisq
      chisq=np.dot(np.transpose(difference),np.dot(self.inv_covmat,difference))

     # correct eventually for effect of analytic marginalization over nuisance parameters
      if ((self.calib_uncertainty > 1.e-4) or self.has_beam_uncertainty):
            
        denom=1.                  
        tmp= np.dot(self.inv_covmat,theo)                             
        chi2op=np.dot(np.transpose(difference),tmp)
        chi2pp=np.dot(np.transpose(theo),tmp)
         
        if (self.has_beam_uncertainty):
          for points in range(self.num_points):
            beam[point]=self.beam_error[point]*theo[point]
          tmp=np.dot(self.inv_covmat,beam)
          chi2dd=np.dot(np.transpose(beam),tmp)
          chi2pd=np.dot(np.transpose(theo),tmp)
          chi2od=np.dot(np.transpose(difference),tmp)
            
        if (self.calib_uncertainty > 1.e-4):
          wpp=1/(chi2pp+1/self.calib_uncertainty**2)
          chisq=chisq-wpp*chi2op**2
          denom = denom/wpp*self.calib_uncertainty**2
        else:
          wpp=0

        if (self.has_beam_uncertainty):
          wdd=1/(chi2dd-wpp*chi2pd**2+1)
          chisq=chisq-wdd*(chi2od-wpp*chi2op*chi2pd)**2
          denom=denom/wdd

        chisq+=log(denom)

   # finally, return ln(L)=-chi2/2

    self.lkl = - 0.5 * chisq 
    return self.lkl



###################################
# CLIK TYPE LIKELIHOOD
# --> clik_fake_planck,clik_wmap,etc.
###################################
class likelihood_clik(likelihood):

  def __init__(self,path,data,command_line,log_flag,default):

    likelihood.__init__(self,path,data,command_line,log_flag,default)
    self.need_cosmo_arguments(data,{'lensing':'yes', 'output':'tCl lCl pCl'})

    if not default:
      return

    try:
      import clik
    except ImportError:
      print " /|\  You must first activate the binaries from the Clik distribution,"
      print "/_o_\ please run : source /path/to/clik/bin/clik_profile.sh"
      print "      and try again."
      exit()
    self.clik = clik.clik(self.path_clik)
    self.l_max = max(self.clik.get_lmax())
    self.need_cosmo_arguments(data,{'l_max_scalars':self.l_max})

    # deal with nuisance parameters
    try:
      self.use_nuisance
    except:
      self.use_nuisance = []
    self.read_contamination_spectra(data)

  def loglkl(self,_cosmo,data):

    nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])

    # get Cl's from the cosmological code
    cl = self.get_cl(_cosmo)
 
    # add contamination spectra multiplied by nuisance parameters
    cl = self.add_contamination_spectra(cl,data)

    # allocate array of Cl's and nuisance parameters
    tot=np.zeros(np.sum(self.clik.get_lmax())+6+len(self.clik.get_extra_parameter_names()))

    # fill with Cl's
    index=0
    for i in range(np.shape(self.clik.get_lmax())[0]):
      if (self.clik.get_lmax()[i] >-1):
        for j in range(self.clik.get_lmax()[i]+1):
          if (i==0):
            tot[index+j]=cl['tt'][j]        
          if (i==1):
            tot[index+j]=cl['ee'][j]
          if (i==2):
            tot[index+j]=cl['bb'][j]
          if (i==3):
            tot[index+j]=cl['te'][j]
          if (i==4):
            tot[index+j]=cl['tb'][j]
          if (i==5):
            tot[index+j]=cl['eb'][j]

        index += self.clik.get_lmax()[i]+1

    # fill with nuisance parameters
    for nuisance in self.clik.get_extra_parameter_names():
      if nuisance in nuisance_parameter_names:
	nuisance_value = data.mcmc_parameters[nuisance]['current']*data.mcmc_parameters[nuisance]['initial'][4]
      else:
	print 'the likelihood needs a parameter '+nuisance
	print 'you must pass it through the input file (as a free nuisance parameter or a fixed parameter)'
	exit()
      tot[index]=nuisance_value
      index += 1

    # compute likelihood
    lkl=self.clik(tot)[0]

    # add prior on nuisance parameters
    lkl = self.add_nuisance_prior(lkl,data)

    return lkl

###################################
# MOCK CMB TYPE LIKELIHOOD
# --> mock planck, cmbpol, etc.
###################################
class likelihood_mock_cmb(likelihood):

  def __init__(self,path,data,command_line,log_flag,default):

    likelihood.__init__(self,path,data,command_line,log_flag,default)

    self.need_cosmo_arguments(data,{'lensing':'yes', 'output':'tCl lCl pCl'})
    
    if not default:
      return
      
    ################
    # Noise spectrum 
    ################

    # convert arcmin to radians
    self.theta_fwhm *= np.array([math.pi/60/180])
    self.sigma_T *= np.array([math.pi/60/180])
    self.sigma_P *= np.array([math.pi/60/180])

    # compute noise in muK**2
    self.noise_T=np.zeros(self.l_max+1,'float64')    
    self.noise_P=np.zeros(self.l_max+1,'float64')

    for l in range(self.l_min,self.l_max+1):
      self.noise_T[l]=0
      self.noise_P[l]=0
      for channel in range(self.num_channels):
        self.noise_T[l] += self.sigma_T[channel]**-2*math.exp(-l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
        self.noise_P[l] += self.sigma_P[channel]**-2*math.exp(-l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
      self.noise_T[l]=1/self.noise_T[l]
      self.noise_P[l]=1/self.noise_P[l]

    # impose that the cosmological code computes Cl's up to maximum l needed by
    # the window function
    self.need_cosmo_arguments(data,{'l_max_scalars':self.l_max})

    ###########
    # Read data
    ###########

    # If the file exists, initialize the fiducial values
    self.Cl_fid = np.zeros((3,self.l_max+1),'float64')
    self.fid_values_exist = False
    if os.path.exists(self.data_directory+'/'+self.fiducial_file):
      self.fid_values_exist = True
      fid_file = open(self.data_directory+'/'+self.fiducial_file,'r')
      line = fid_file.readline()
      while line.find('#')!=-1:
	line = fid_file.readline()
      while (line.find('\n')!=-1 and len(line)==1):
	line = fid_file.readline()
      for l in range(self.l_min,self.l_max+1):
        ll=int(line.split()[0])
        self.Cl_fid[0,ll]=float(line.split()[1])
        self.Cl_fid[1,ll]=float(line.split()[2])
        self.Cl_fid[2,ll]=float(line.split()[3])
        line = fid_file.readline()

    # Else the file will be created in the loglkl() function. 

    # end of initialisation
    return

  def loglkl(self,_cosmo,data):

    # get Cl's from the cosmological code (returned in muK**2 units)
    cl = self.get_cl(_cosmo)

    # get likelihood
    lkl = self.compute_lkl(cl,_cosmo,data)

    return lkl

  def compute_lkl(self,cl,_cosmo,data):

    # Write fiducial model spectra if needed (exit in that case)
    if self.fid_values_exist is False:
      # Store the values now, and exit.
      fid_file = open(self.data_directory+'/'+self.fiducial_file,'w')
      fid_file.write('# Fiducial parameters')
      for key,value in data.mcmc_parameters.iteritems():
	fid_file.write(', %s = %.5g' % (key,value['current']*value['initial'][4]))
      fid_file.write('\n')
      for l in range(self.l_min,self.l_max+1):
        fid_file.write("%5d  " % l)
        fid_file.write("%.8g  " % (cl['tt'][l]+self.noise_T[l]))
        fid_file.write("%.8g  " % (cl['ee'][l]+self.noise_P[l]))
        fid_file.write("%.8g  " % cl['te'][l])
        fid_file.write("\n")
      print '\n\n /|\  Writting fiducial model in {0}'.format(self.data_directory+self.fiducial_file)
      print '/_o_\ for {0} likelihood'.format(self.name)
      return 1

    # compute likelihood

    chi2=0

    Cov_obs=np.zeros((2,2),'float64')
    Cov_the=np.zeros((2,2),'float64')
    Cov_mix=np.zeros((2,2),'float64')

    for l in range(self.l_min,self.l_max+1):

      Cov_obs=np.array([[self.Cl_fid[0,l],self.Cl_fid[2,l]],[self.Cl_fid[2,l],self.Cl_fid[1,l]]])
      Cov_the=np.array([[cl['tt'][l]+self.noise_T[l],cl['te'][l]],[cl['te'][l],cl['ee'][l]+self.noise_P[l]]])

      det_obs=np.linalg.det(Cov_obs) 
      det_the=np.linalg.det(Cov_the)
      det_mix=0.

      for i in range(2):
	Cov_mix = np.copy(Cov_the)
        Cov_mix[:,i] = Cov_obs[:,i]
	det_mix += np.linalg.det(Cov_mix)

      chi2 += (2.*l+1.)*self.f_sky*(det_mix/det_the + math.log(det_the/det_obs) - 2)

    return -chi2/2

###################################
# MPK TYPE LIKELIHOOD
# --> sdss, wigglez, etc.
###################################
class likelihood_mpk(likelihood):

  def __init__(self,path,data,command_line,log_flag,default):

    likelihood.__init__(self,path,data,command_line,log_flag,default)

    # require P(k) from class
    self.need_cosmo_arguments(data,{'output':'mPk'})

    try:
      self.use_halofit
    except:
      self.use_halofit = False

    if self.use_halofit:
      self.need_cosmo_arguments(data,{'non linear':'halofit'}) 

    # read values of k (in h/Mpc)
    self.k_size=self.max_mpk_kbands_use-self.min_mpk_kbands_use+1
    self.mu_size=1
    self.k = np.zeros((self.k_size),'float64')
    self.kh = np.zeros((self.k_size),'float64')

    datafile = open(self.data_directory+self.kbands_file,'r')
    for i in range(self.num_mpk_kbands_full):
      line = datafile.readline()
      if i+2 > self.min_mpk_kbands_use and i < self.max_mpk_kbands_use:
        self.kh[i-self.min_mpk_kbands_use+1]=float(line.split()[0])
    datafile.close()      

    khmax = self.kh[-1]

    # check if need hight value of k for giggleZ
    try:
      self.Use_giggleZ
    except:
      self.Use_giggleZ = False

    if self.Use_giggleZ:
      datafile = open(self.data_directory+self.giggleZ_fidpk_file,'r')

      line = datafile.readline()
      k=float(line.split()[0])
      line_number=1
      while (k<self.kh[0]):
        line = datafile.readline()
        k=float(line.split()[0])
        line_number += 1
      ifid_discard=line_number-2  
      while (k<khmax):
        line = datafile.readline()
        k=float(line.split()[0])
        line_number += 1
      datafile.close()
      self.k_fid_size=line_number-ifid_discard+1  
      khmax=k

    if self.use_halofit:  
      khmax *= 2

    # require k_max and z_max from the cosmological module
    self.need_cosmo_arguments(data,{'P_k_max_h/Mpc':khmax,'z_max_pk':self.redshift})

    # In case of a comparison, stop here
    if not default:
      return

    # read information on different regions in the sky
    try:
      self.has_regions
    except:
      self.has_regions = False

    if (self.has_regions):
      self.num_regions = len(self.used_region)
      self.num_regions_used = 0
      for i in range(self.num_regions):
        if (self.used_region[i]):
          self.num_regions_used += 1
      if (self.num_regions_used == 0):
        print 'mpk: no regions begin used in this data set'
        exit()
    else:
      self.num_regions = 1
      self.num_regions_used = 1
      self.used_region = [True]
 
    # read window functions
    self.n_size=self.max_mpk_points_use-self.min_mpk_points_use+1

    self.window=np.zeros((self.num_regions,self.n_size,self.k_size),'float64')

    datafile = open(self.data_directory+self.windows_file,'r')
    for i_region in range(self.num_regions):
      if self.num_regions>1:
        line = datafile.readline()
      for i in range(self.num_mpk_points_full):
        line = datafile.readline()
        if i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use:
          for j in range(self.k_size):  
            self.window[i_region,i-self.min_mpk_points_use+1,j]=float(line.split()[j+self.min_mpk_kbands_use-1])
    datafile.close()    

    # read measurements
    self.P_obs=np.zeros((self.num_regions,self.n_size),'float64')
    self.P_err=np.zeros((self.num_regions,self.n_size),'float64')

    datafile = open(self.data_directory+self.measurements_file,'r')
    for i_region in range(self.num_regions):
      for i in range(2):
        line = datafile.readline()
      for i in range(self.num_mpk_points_full):
        line = datafile.readline()
        if i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use:
          self.P_obs[i_region,i-self.min_mpk_points_use+1]=float(line.split()[3])  
          self.P_err[i_region,i-self.min_mpk_points_use+1]=float(line.split()[4])  
    datafile.close()

    # read covariance matrices
    try:
      self.covmat_file
      self.use_covmat = True
    except:
      self.use_covmat = False

    self.invcov=np.zeros((self.num_regions,self.n_size,self.n_size),'float64')

    if self.use_covmat:  
      cov = np.zeros((self.n_size,self.n_size),'float64')
      invcov_tmp = np.zeros((self.n_size,self.n_size),'float64')

      datafile = open(self.data_directory+self.covmat_file,'r')
      for i_region in range(self.num_regions):
        for i in range(1):
          line = datafile.readline()
        for i in range(self.num_mpk_points_full):
          line = datafile.readline()
          if i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use:
            for j in range(self.num_mpk_points_full):
              if j+2 > self.min_mpk_points_use and j < self.max_mpk_points_use:
                cov[i-self.min_mpk_points_use+1,j-self.min_mpk_points_use+1]=float(line.split()[j])
        invcov_tmp=np.linalg.inv(cov)
        for i in range(self.n_size):
          for j in range(self.n_size):
            self.invcov[i_region,i,j]=invcov_tmp[i,j]
      datafile.close()
    else:
      for i_region in range(self.num_regions):
        for j in range(self.n_size):
          self.invcov[i_region,j,j]=1./(self.P_err[i_region,j]**2)
          

    # read fiducial model
    if self.Use_giggleZ:
      self.P_fid=np.zeros((self.k_fid_size),'float64')
      self.k_fid=np.zeros((self.k_fid_size),'float64')
      datafile = open(self.data_directory+self.giggleZ_fidpk_file,'r')
      for i in range(ifid_discard):
        line = datafile.readline()
      for i in range(self.k_fid_size):
        line = datafile.readline()
        self.k_fid[i]=float(line.split()[0])
        self.P_fid[i]=float(line.split()[1])
      datafile.close()

    # assign defaut value to optional parameters not being in the .data
    try:
      self.Use_jennings
    except:
      self.Use_jennings = False

    try:
      self.Use_simpledamp
    except:
      self.Use_simpledamp = False

    return

  # compute likelihood

  def loglkl(self,_cosmo,data):

    # reduced Hubble parameter
    h = _cosmo._h()
     
    if self.use_scaling:
      # angular diameter distance at this redshift, in Mpc/h
      d_angular = _cosmo._angular_distance(self.redshift)
      d_angular *= h  

      # radial distance at this redshift, in Mpc/h
      r,Hz = _cosmo.z_of_r([self.redshift])
      d_radial = self.redshift*h/Hz[0]

      # scaling factor = (d_angular**2 * d_radial)^(1/3) relative
      # to a fiducial model
      scaling = pow((d_angular/self.d_angular_fid)**2*(d_radial/self.d_radial_fid),1./3.)
    else:
      scaling = 1

    # get rescaled values of k in 1/Mpc
    self.k = self.kh *h*scaling

    # get P(k) at right values of k, convert it to (Mpc/h)^3 and rescale it
    P_lin = np.zeros((self.k_size),'float64')

    if self.Use_giggleZ:

      P = np.zeros((self.k_fid_size),'float64')  
      
      for i in range(self.k_fid_size):

        P[i] = _cosmo._pk(self.k_fid[i]*h,self.redshift)

        power=0
        for j in range(6):
          power += self.giggleZ_fidpoly[j]*self.k_fid[i]**j

        # rescale P by fiducial model and get it in (Mpc/h)**3
        P[i] *= pow(10,power)/self.P_fid[i]*(h/scaling)**3

      # get rescaled values of k in 1/Mpc
      #self.k=self.kh *h*scaling

      # get P_lin by interpolation. It is still in (Mpc/h)**3
      P_lin = np.interp(self.kh,self.k_fid,P)

    else:
      # get rescaled values of k in 1/Mpc
      self.k=self.kh *h*scaling
      # get values of P(k) in Mpc**3
      for i in range(self.k_size):
        P_lin[i] = _cosmo._pk(self.k[i],0)
      # get rescaled values of P(k) in (Mpc/h)**3
      P_lin *= (h/scaling)**3

    do_marge = self.Q_marge

    W_P_th =  np.zeros((self.n_size),'float64')

    if do_marge and self.Q_flat:

      P_th =  np.zeros((self.k_size),'float64')
      for i in range(self.k_size):
        P_th[i] = P_lin[i]/(1.+self.Ag*self.kh[i]) 

      k2 =  np.zeros((self.k_size),'float64')
      for i in range(self.k_size):
        k2[i] = P_th[i] * self.kh[i]**2

      W_P_th_k2 =  np.zeros((self.n_size),'float64')  
      covdat = np.zeros((self.n_size),'float64') 
      covth = np.zeros((self.n_size),'float64') 
      covth_k2 = np.zeros((self.n_size),'float64') 

      chi2 = 0
      for i_region in range(self.num_regions):
        if self.used_region[i_region]:
          W_P_th = np.dot(self.window[i_region,:],P_th)
          W_P_th_k2 = np.dot(self.window[i_region,:],k2)
          
          covdat  =np.dot(self.invcov[i_region,:,:],self.P_obs[i_region,:])
          covth   =np.dot(self.invcov[i_region,:,:],W_P_th)
          covth_k2=np.dot(self.invcov[i_region,:,:],W_P_th_k2)

          offdiag=sum(covth*W_P_th_k2)

          Mat=np.zeros((2,2),'float64')
          Mat=[[sum(covth*W_P_th),offdiag],[offdiag,sum(covth_k2*W_P_th_k2)]]

          Vec=np.zeros((2),'float64')
          Vec=[sum(covdat*W_P_th),sum(covdat*W_P_th_k2)]

          chi2+=-sum(self.P_obs[i_region,:]*covdat)+np.dot(Vec,np.dot(np.linalg.inv(Mat),Vec))-math.log(np.linalg.det(Mat))

      return -chi2/2

    else:  

      if (self.Q_sigma == 0):
        do_marge = False

      if (self.Use_jennings or self.Use_simpledamp):
        print "case with Use_jennings or Use_simpledamp not coded"
        exit()
      else :
        #starting analytic marginalisation over bias

        P_data_large =  np.zeros((self.n_size*self.num_regions_used),'float64')
        W_P_th_large =  np.zeros((self.n_size*self.num_regions_used),'float64')
        cov_dat_large =  np.zeros((self.n_size*self.num_regions_used),'float64')
        cov_th_large =  np.zeros((self.n_size*self.num_regions_used),'float64')

        normV = 0

        if do_marge:
          nQ = 6
          dQ = 0.4
        else:
          nQ=0
          dQ=0

        chisq = np.zeros((nQ*2+1),'float64')
        calweights = np.zeros((nQ*2+1),'float64')

        for iQ in range(-nQ,nQ+1):
          
          # infer P_th from P_lin. It is still in (Mpc/h)**3
          P_th =  np.zeros((self.k_size),'float64')
          for i in range(self.k_size):
            if self.Q_marge:
              Q = self.Q_mid +iQ*self.Q_sigma*dQ 
              P_th[i] = P_lin[i]*(1+Q*self.kh[i]**2)/(1.+self.Ag*self.kh[i]) 
            else:
              P_th[i] = P_lin[i]

          for i_region in range(self.num_regions):
            if self.used_region[i_region]:
              imin = i_region*self.n_size
              imax = (i_region+1)*self.n_size-1

              W_P_th = np.dot(self.window[i_region,:],P_th)
              for i in range(self.n_size):
                P_data_large[imin+i] = self.P_obs[i_region,i]
                W_P_th_large[imin+i] = W_P_th[i]
                cov_dat_large[imin+i] = np.sum(self.invcov[i_region,i,:]*self.P_obs[i_region,:])
                cov_th_large[imin+i] = np.sum(self.invcov[i_region,i,:]*W_P_th[:])
          normV += np.sum(W_P_th_large*cov_th_large)
          b_out = np.sum(W_P_th_large*cov_dat_large)/np.sum(W_P_th_large*cov_th_large)
          #print "bias value",b_out
          chisq[iQ+nQ] = np.sum(P_data_large*cov_dat_large)  - np.sum(W_P_th_large*cov_dat_large)**2/normV
          
          if do_marge:
            calweights[iQ+nQ] = exp(-(iQ*dQ)**2/2)
          else: 
            return -chisq[iQ+nQ]/2

      if do_marge:
        if not self.Use_jennings:
          minchisq=np.min(chisq)
          lnlike = np.sum(exp(-(chisq[:]-minchisq)/2)*calweights[:])/np.sum(calweights[:])
          if (lnlike == 0):
            return data.boundary_loglike
          else:
            print "case with Use_jennings not coded"
            exit()

