"""
Code to test SPRT versus averaging loss function samples!
@author: phillips_pc
"""

import numpy as np
from time import time 
from numpy import random as rand  
from scipy import linalg
#from numba import prange, jit
#averaging schedule. May want to make more sophisticated. 
def avgnum(k):
    return 5;
#declare loss function 
def loss(x, sigma): 
    #should be able to handle any dimension
    #needs to return scalar values 
    #diving norm by 10 to make problem a bit harder
    return np.sqrt(linalg.norm(x))/10+sigma*rand.randn()

#script to be able to test the SPRT search
#as general as possible, to easily use new functions 
#and easily use new search spaces 


def main():
    #declare constants
    xdim = 2        #set dimension of search space 
    discrete = True #discrete or continuous space?
    N = 100        #will search in box or grid of [-N, N]
    runs = 20       #number of runs to average performance 
    evals = 10000    #number of function evaluations allowed 
    eta = 1/10       #min difference in loss values for distinct x (epsilon in report) 
    
    sigma = np.float64(1)       #noise parameter 
    
    
    #SPRT hyperparameters 
    maxdraws = evals  #optional argument to allow a max draw number, or set to evals
    alpha = 0.1   #alpha value (fixed here for simplicity, but may want to take descreasing sequence instead!)
    beta = 0.3   #beta value 
    
    A = np.log((1-beta)/alpha) #corresponding A, B values 
    B = np.log(beta/(1-alpha))
    
    #data containers for terminal values 
    termloss_sprt = np.empty(runs)
    termloss_avg = np.empty(runs)
    
    for r in range(runs):
        #generate candidate xvals first to have CRN's 
        xcands = np.float64(gen_cands(evals, N, discrete, xdim))
        xcount = 0
        xnow = xcands[xcount, :]
        xcand = np.copy(xnow) #just so my IDE doesnt complain later...
        
        k = 0
        accepted = False;
        
        #implement SPRT random search 
        while k <= evals:
            if accepted: 
                xnow = xcand
            #else just do nothing 
            
            xcount += 1
            xcand = xcands[xcount, :] #new candidate x point 
            
            #do SPRT 
            accepted, draws = SPRT(xnow, xcand, eta, sigma, maxdraws, A, B)
            k += 2*draws
            
        if accepted: #just need to check ~one more time~
            xnow = xcand
        termloss_sprt[r] = loss(xnow, 0)
        print(xnow)
        
        ##########################
        #Now do averaging method 
        xcount = 0
        xnow = xcands[xcount, :]
        xcand = np.copy(xnow) #just so my IDE doesnt complain later...
        
        k = 0
        accepted = False
        
        while k <= evals:
            xcount += 1;
            xcand = xcands[xcount, :]
            
            avj = avgnum(k)
            nowmean = 0
            candmean = 0
            
            for p in range(avj):
                nowmean += loss(xnow, sigma) #can skip the /avj actually...
                candmean += loss(xcand, sigma)
                
            if candmean < nowmean:
                xnow = xcand
            #else go to next iteration of loop
            k+= avj;
        termloss_avg[r]= loss(xnow, 0);
        
    print('Mean of terminal SPRT loss:', np.mean(termloss_sprt))
    print('Mean of terminal AVG  loss:', np.mean(termloss_avg))
    
    return termloss_sprt, termloss_avg, xcands


def SPRT(xnow, xcand, eta, sigma, maxdraws, A, B):
#perform SPRT with xnow vs xcount 
    #fxnows = np.array([])
    #fxcands = np.array([]) #may want to keep track for sample mean?
    
    zsum = 0
    for j in range(maxdraws):
        Yxnow = loss(xnow, sigma)
        Yxcand= loss(xcand, sigma)
        
        Z = Yxnow - Yxcand
        #H0 is E(Z) < -eta/2, for now. But should it be??
        #note that if evaluating normpdf we could get 0 numerically 
        #that's okay since then we have overwhelming evidence!!
        #note that accepting H0 is rejecting the new point
        #hence accepted is set to True when H0 is rejected and vice versa 
    
        f0 = normpdf(Z, 0, sigma)
        f1 = normpdf (Z, -eta, sigma) #this is super conservative  
        #deal with potential numerical problems here 
        if f0 == 0 and Z < 0:
            return False, (j+1)
        if f1 == 0 and Z > 0:
            return True, (j+1) 
        
        zsum += np.log(f1/f0) #increment sum 
        if zsum >= A:
            return False, (j+1)
        if zsum <= B: 
            return True, (j+1)
        if j == (maxdraws-1):
            return True, (j+1)





#function to generate new x candidates 
#as an array. Either in integer grid or in square 
#generating as array to easily use CRN's 
def gen_cands(evals, N, discrete, xdim):
    if discrete:
        return rand.randint(-N, N+1, (evals, xdim))
    else: 
        return (2*N*rand.rand(evals, xdim)-N)

#function to just evaluate normal density
#evaluate at x with parameters mu, sigma  
def normpdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2))

t0 = time()
termloss_sprt, termloss_avg, xcands = main()
print('runtime was', time()-t0)
