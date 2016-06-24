#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import emcee
from scipy import pi,sqrt,exp,interpolate
from scipy.special import erf
from astropy.table import Table
import itertools
import sys,time,datetime, math,re,operator
from datetime import datetime

'''
CASE I: calculating the stellar parameters based on suspected Cycle 12 ACIS calibration

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
        c = Table.read('/Users/Phill/Astro/ACIS_Cal/V2/channel_spectrum.fits',format='fits')
        self.c = c[(c['pi'] > minchannel) & (c['pi'] <= maxchannel)]

        # dummy channel array, this is the range the calibration is calculated over, so it is important
        self.chan = np.arange(1,1025,1)
        self.good_values = [(self.chan>minchannel) & (self.chan<=maxchannel)]

        # parameters for Black Body
        # make a dummy energy array from 0.175 to 0.31 
        self.e = np.arange(0.175,0.31,0.01)
        self.Lstar = np.power(10,2.49)*3.8e+33
        
        self.Teff = 108600.0
        self.Rad  = 0.045
        self.Dist = 405.0/1000.0
        self.NH = 0.05
        
        self.eTeff = 6800.0
        self.eRad  = 0.004
        self.eDist = 28.0/1000.0
        self.eNH = 0.01
        
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
        Lstar = (Rad**2.0)*((Teff/5770.0)**4.0)*3.8e+33

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

    def logprior(self,t):
        ''' 
        In CASE I, there are only priors on the stellar parameters: 
        '''
        # free parameters
        Teff,Dist,Rad,NH = t

        if not ((NH >= 0.0) & (NH < 0.1)): # priors are absorption (cm^-3)
            return -np.inf

        if not ((Teff > 60000.0) & (Teff < 200000.0)):  # priors on Teff (K)
            return -np.inf

        if not ((Rad > 0.0) & (Rad < 0.5)): # priors are stellar radius (R_sun)
            return -np.inf

        if not ((Dist > 0.0) & (Dist < 1.0)): # priors are distance (kpc)
            return -np.inf

        return 0.0

    def loglhood(self,t):

        # Free parameters, for case 1: Teff, distance, radius
        Teff,Dist,Rad,NH = t

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
                arf[ii] = self.arf1
            elif (ee_i >= 0.23) and (ee_i < 0.26):
                arf[ii] = self.arf2
            elif (ee_i >= 0.26) and (ee_i < 0.287):
                arf[ii] = self.arf3
            elif (ee_i >= 0.287):
                arf[ii] = self.arf4


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
        lp = self.logprior(par)
        if (lp == -np.inf):
            return lp
        return lp + self.loglhood(par)

    def run_MCMC(self):
        # Now, set up and run the sampler:

        # number of walkers
        nwalkers = 250

        # initial ball values for free parameters
        # Tefff, Dist, Rad, N_H
        p0mean = [143431.928537, 0.410281810919, 0.0333451376657, 0.05]
        p0std  = [2000.0,        0.01,           0.01,            0.001]

        # set up initial sampler's ball positions
        p0 = [[y*np.random.triangular(-1,0,1)+x for x,y in zip(p0mean,p0std)] for _ in range(nwalkers)]

        # number of dim
        ndim = len(p0mean)

        # Instantiate the class
        likefunc = self.calllike

        # The sampler object:
        # sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, threads=10)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likefunc, threads=0)

        # Burn in.
        startbin = datetime.now()
        bintime = startbin
        iternum = 1
        for pos0, prob0, state0 in sampler.sample(p0,iterations=500,storechain=False):
            if iternum % 100 == 0.0:
                print('BURN IN: Finished iteration: {0} -- Time: {1} -- mean(AF): {2}'.format(
                    iternum,datetime.now()-bintime,np.mean(sampler.acceptance_fraction)))
                bintime = datetime.now()
            iternum = iternum + 1

        # Clear the burn in.
        sampler.reset()

        # Sample, outputting to a file
        fn = "ACIScal_I.out"

        with open(fn, "w") as f:
            f.write('WN Teff Dist Rad NH lnprob\n')

        walkernum = np.arange(1,nwalkers+1,1).reshape(nwalkers,1)
        startmct = datetime.now()
        ii = 1
        for pos, prob, rstate in sampler.sample(pos0, prob0, state0, iterations=1000,storechain=False):
            pos_matrix=pos.reshape(nwalkers,ndim)
            pos_matrix=np.append(walkernum,pos_matrix,axis=1)
            prob_array=prob.reshape(nwalkers,1)
            steparray =np.append(pos_matrix,prob_array,axis=1)
            if ii % 100 == 0.0:
                print('Finished iteration: {0} -- Time: {1} -- mean(AF): {2}'.format(
                                    ii,datetime.now()-startmct,np.mean(sampler.acceptance_fraction)))
                startmct = datetime.now()
            # Write the current position to a file, one line per walker
            f = open(fn, "a")
            f.write("\n".join(["\t".join([str(q) for q in p]) for p in steparray]))
            f.write("\n")
            f.close()
            ii = ii + 1
        t = Table.read(fn,format='ascii')
        t.write(fn+'.fits',format='fits')
        print('Total Time -> {0}'.format(datetime.now()-startbin))        

if __name__ == '__main__':
    # initialize the class
    c = ACIS()
    # run the MCMC
    c.run_MCMC()
