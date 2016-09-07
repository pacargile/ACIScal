#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from scipy import pi,sqrt,exp,interpolate
from scipy.special import erf,ndtri
from astropy.table import Table
import itertools
import sys,time,datetime, math,re,operator
from datetime import datetime
import nestle
import h5py

'''
CASE Vb: NEEDS UPDATE

calculating the stellar parameters based on suspected Cycle 12 ACIS calibration
 -> Ib : Using gaussian prior on distance based on the measured parallax of the star

Generally, we use specific energy bins to generate the effective area curve, the bins are: 
e1 = 0.175 - 0.231 keV,  e2 = 0.231 - 0.258 keV,  e3 = 0.258 - 0.277 keV,  e4 = 0.277 - 0.3 keV
this bins are set in the likelihood call because eventually they will be free parameters, but  
in CASE I, they are set to the best guess from CXC Cycle 12. 

For this case, we have some priors on the stellar parameters. 

'''

# Define the posterior density to be sampled:
class ACIS(object):
    def __init__(self):

        # set energy channels used in calculation
        minchannel = 12
        maxchannel = 20

        # load the channel spectrum (PI):
        # using astropy
        c = Table.read('channel_spectrum.fits',format='fits')
        self.c = c[(c['pi'] > minchannel) & (c['pi'] <= maxchannel)]

        # dummy channel array, this is the range the calibration is calculated over, so it is important
        self.chan = np.arange(1,1025,1)
        self.good_values = [(self.chan>minchannel) & (self.chan<=maxchannel)]

        # parameters for Black Body
        # make a dummy energy array from 0.175 to 0.31 
        self.e = np.arange(0.175,0.31,0.01)
        self.Lstar = np.power(10,2.49)*3.828e+33
        
        self.Teff = 108600.0
        self.Rad  = 0.045
        self.Dist = 405.0/1000.0
        Av = 0.30
        self.NH = Av*(1.79E+21)/(1E+22)
        
        self.eTeff = 6800.0
        self.eRad  = 0.004
        self.eDist = 28.0/1000.0
        self.eNH = 0.01
        
        # CASE I number of dimensions
        self.ndim = 6

        # functional form of exinction curve, W(N_H), from source unknown
        c0 = [34.6,267.9,-476.1]
        c1 = [78.1, 18.8, 4.3]
        e0 = np.where((self.e>0.1)&(self.e<=0.284))
        e1 = np.where((self.e>0.284)&(self.e<=0.4))
        w = np.zeros_like(self.e)
        w[e0] = 1e-2*(c0[0]+c0[1]*self.e[e0]+c0[2]*self.e[e0]**2)/self.e[e0]**3
        w[e1] = 1e-2*(c1[0]+c1[1]*self.e[e1]+c1[2]*self.e[e1]**2)/self.e[e1]**3
        self.w = w
        
        # original initial values for RMF
        self.initarr = ({
            'mu_m':74.3577,
            'mu_b':-0.2841,
            'sigma_m':3.3077,
            'sigma_b':2.5584,
            'alpha_pi':-1.9126,
            'sigma':0.05951348,
            'mu':0.28797341,
            'alpha_arf':-16.19973467,
            'a':7.7096437/8.0
            })

        # the following is for when piprob is fixed
        self.fixpiprob = ([
            2.0*self.NormPDF(self.chan,mu_i,sig_i)*self.NormCDF(self.chan,mu_i,sig_i,self.initarr['alpha_pi']) 
            for mu_i,sig_i in zip(self.initarr['mu_m']*self.e+self.initarr['mu_b'],
                self.initarr['sigma_m']*self.e+self.initarr['sigma_b'])
            ])

        # Define what Chandra thinks the ARF weights should be
        self.arf1 = 40.0
        self.arf2 = 75.0
        self.arf3 = 100.0
        self.arf4 = 5.0

    def wabs(self,nh): # c=[34.6,267.9,-476.1]):
        ''' wabs function used to calculate extinction for a given nh and using the functional form stored in self.w '''
        return np.exp(-nh*self.w)
    def NormPDF(self,x,mu,sigma):
        ''' NormPDF used in piprob function '''
        return (1/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x-mu)/(np.sqrt(2.0)*sigma))**2.0)
    def NormCDF(self,x,mu,sigma,alpha):
        ''' NormCDF used in piprob function '''
        return (0.5*(1.0+erf((alpha*((x-mu)/sigma))/np.sqrt(2.0))))
    def bb_free(self,Teff,Dist,Rad):
        """ 
        bb function is the blackbody source function, for M27, with the values given below.
        """
        Lstar = (Rad**2.0)*((Teff/5777.0)**4.0)*3.828e+33

        BBnorm = (Lstar/1.0e+39)*np.power(Dist/10.0,-2.0)
        kT = Teff/11604.5e+3 
        return BBnorm*8.0525*np.power(self.e,2.0)*np.power(kT,-4.0)/(np.exp(self.e/kT)-1.0)
    def piprob(self,energyspectrum,p0):
        """
        piprob returns the skewed gaussian probability distribution for a given energy. 
        This is purely due to the detector.

        The behavior was determined from extrapolation of the RMF behavior above 0.3 keV.
        The sigma and mu for the gaussian are linear extrapolations while the alpha or 
        skewness parameter is assumed to plateau at -1.9126. 
        
        An alternative extrapolation would be for sigma to also plateau at a value of 3.6.
        
        In the future, or in the MCMC, maybe we could vary the sigma... probably not.
        """
        mu_m,mu_b,sigma_m,sigma_b,alpha_pi = p0
        mu = mu_m*self.e+mu_b
        sigma = sigma_m*self.e+sigma_b
        cs = np.zeros(len(self.chan)).astype(np.float64)
        # each energy bin of photons arriving at the detector (ARF*S) are spread onto the channels:
        # uncomment following when piprob has free parameters
        # for mu_i,sig_i,es_i in zip(mu,sigma,energyspectrum):
        #     cs += 2.0*self.NormPDF(self.chan,mu_i,sig_i)*self.NormCDF(self.chan,mu_i,sig_i,alpha_pi)*es_i
        
        # uncomment following when piprob is fixed
        for ii,es_i in enumerate(energyspectrum):
            cs += self.fixpiprob[ii]*es_i
        
        return cs
   
    def prior_trans(self,t):
        Teff,Dist,Rad,NH,arfsc1,arfsc2 = t

        # define range and min values
        Teffrange = 200000.0-60000.0
        Teffmin   = 60000.0

        # Distrange = 1.0-0.0
        # Distmin   = 0.0

        Distmu = 405.0/1000.0
        Distsig = 28.0/1000.0

        Radrange = 0.08-0.0
        Radmin   = 0.0

        NHrange = 0.1-0.0
        NHmin   = 0.0

        arfscran = 1.0 - 0.0
        arfscmin = 0.0

        # build normalized prior array
        outarr = np.array([
            Teffrange*Teff+Teffmin,
            # Distrange*Dist+Distmin,
            Distmu + Distsig*ndtri(Dist),
            Radrange*Rad+Radmin,
            NHrange*NH+NHmin,
            arfscran * arfsc1 + arfscmin,
            arfscran * arfsc2 + arfscmin,
            ])
        return outarr

    def loglhood(self,t):

        # Free parameters, for case 1: Teff, distance, radius
        Teff,Dist,Rad,NH,arfsc1,arfsc2 = t

        # calculate blackbody
        bbody_i = self.bb_free(Teff,Dist,Rad)

        # generate N_H function
        # extarr = np.ones_like(bbody_i)
        extarr = self.wabs(NH)

        # create extincted blackbody
        bbody = np.multiply(bbody_i,np.array(extarr))

        # produce a dummy array for arf
        arf = np.ones_like(bbody)

        # populate the arf array from best-guest (CASE I) for the 4 energy bins
        for ii,ee_i in enumerate(self.e):
            if (ee_i >= 0.175) and (ee_i < 0.23):
                arf[ii] = self.arf1*arfsc1
            elif (ee_i >= 0.23) and (ee_i < 0.26):
                arf[ii] = self.arf2*arfsc1
            elif (ee_i >= 0.26) and (ee_i < 0.287):
                arf[ii] = self.arf3*arfsc1
            elif (ee_i >= 0.287):
                arf[ii] = self.arf4*arfsc2


        par_pi = ([self.initarr['mu_m'],self.initarr['mu_b'],self.initarr['sigma_m'],
            self.initarr['sigma_b'],self.initarr['alpha_pi']])

        t_pispec = self.piprob(np.multiply(arf,bbody),par_pi)

        # generate model
        self.mod = (19.5e3)*t_pispec[self.good_values]

        # errors
        err2 = np.array([10.0 for cc in self.c['counts']])

        # create residual^2 / err^2
        resid2 = np.array([ ((m-c)**2.0)/e2 for m,c,e2 in zip(self.mod,self.c['counts'],err2)])
        
        lnprob_out = np.sum(-0.5*resid2 + np.log(1.0/np.sqrt(2*np.pi*(err2))))

        return lnprob_out

    def calllike(self, par):
        # store par into self so that we can write it during callback function
        self.par = par
        # return log(likelihood prob)
        return self.loglhood(par)

    def nestle_callback(self,iterinfo):
        # write iteration number
        self.outfile.write('{0} '.format(iterinfo['it']))
        # write parameters at iteration
        for pp in self.par:
            self.outfile.write('{0} '.format(pp))
        # write the evidence
        self.outfile.write('{0}'.format(iterinfo['logz']))
        # write new line
        self.outfile.write('\n')

        # print iteration number and evidence at specific iterations
        if iterinfo['it'] % 100 == 0:
            if iterinfo['logz'] < -10E+6:
                print('Iter: {0} < -10M'.format(iterinfo['it']))
            else:
                print('Iter: {0} = {1}'.format(iterinfo['it'],iterinfo['logz']))

    def run_nestle(self,outfile='TEST.dat'):
        # initalize outfile 
        self.outfile = open(outfile,'w')
        self.outfile.write('ITER Teff Dist Rad NH arfsc1 arfsc2 log(z) \n')
        # Start sampler
        print('Start Nestle')
        result = nestle.sample(self.calllike,self.prior_trans,self.ndim,method='multi',npoints=1000,callback=self.nestle_callback)
        # generate posterior means and covariances
        p,cov = nestle.mean_and_cov(result.samples,result.weights)
        # close output file
        self.outfile.close()
        return result,p,cov

if __name__ == '__main__':
    # initialize the class
    c = ACIS()
    # outfile names
    outfile = 'ACIScal_I_MNS_CASEVb.out'
    outh5file = 'ACIScal_I_MNS_CASEVb.h5'
    # run the MCMC
    starttime = datetime.now()
    results,p,cov = c.run_nestle(outfile)
    print('FINISHED MNS: {0}s'.format(datetime.now()-starttime))
    print('Summary of MNS... ')
    print(results.summary())

    # write results into HDF5 file
    outh5 = h5py.File(outh5file,'w')
    for kk in results.keys():
        outh5.create_dataset(kk,data=np.array(results[kk]))
    outh5.create_dataset('P',data=np.array(p))
    outh5.create_dataset('COV',data=np.array(cov))

    # flush and close hdf5 file
    outh5.flush()
    outh5.close()
