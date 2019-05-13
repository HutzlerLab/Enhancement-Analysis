'''
Functions useful for curve fitting and related operations.
By Arian Jadbabaie
'''

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import savgol_filter,find_peaks,peak_widths
from math import ceil,floor
from scipy.special import wofz


###############################################################################
'''Functions Involving Curve Fits'''
###############################################################################

def shift3Gaussians(xscale,data):
    mean_guess,stdev_guess,norm_guess = genGuess3Gaussians(xscale,data)
    params_3g,err_3g,resid_3g = fit3Gaussians(xscale, data, mean_guess, stdev_guess, norm_guess,offset_guess=[data.min()],plot=False,verbose=False)
    means = params_3g[3:6]
    means_err = err_3g[3:6]
    #Offset so that the middle peak is at 0
    x_shift = xscale - means[1]
    means_shifted = np.array(means)-means[1]
    return [x_shift,means_shifted,means_err]

def shiftFlatTop(xscale,data,method=1):
    if method == 1:
        loc,width,height = genGuessGaussian(xscale,data)
        guess = [width,loc,height,0,height/1.1]
        peak_idx,properties = find_peaks(data, height =np.amax(data))
        width_info = peak_widths(data,peak_idx)
        width_idx = width_info[0]
        s = slice(int(peak_idx-3*width_idx),int(peak_idx+3*width_idx))
        _x = xscale[s]
        _data = data[s]
        params,error,resid = fitFlatTopGaussian(_x,_data,guess,plot=True,verbose=False)
        mean = params[1]
        x_shift = xscale - mean
    else:
        loc,width,height = genGuessGaussian(xscale,data)
        guess = [width/3,width/3.5,loc,2*height*width,0,0]
        peak_idx,properties = find_peaks(data, height =np.amax(data))
        width_info = peak_widths(data,peak_idx)
        width_idx = width_info[0]
        s = slice(int(peak_idx-3*width_idx),int(peak_idx+3*width_idx))
        _x = xscale[s]
        _data = data[s]
        params,error,resid = fitVoigtLinBG(_x,_data,guess,plot=True,verbose=False)
        mean = params[2]
        x_shift = xscale - mean
    return x_shift


###############################################################################
'''Curve Fitting'''
###############################################################################

def fitFunction(xscale,data,function,guess,sigma,plot,bounds=None):
    xscale = np.array(xscale)
    data = np.array(data)
    guess = np.array(guess).astype(float)
    if bounds==None:
        bounds = (-np.inf,np.inf)
    try:
        popt,pcov = curve_fit(function,xscale,data,p0=guess,sigma=sigma,bounds=bounds)
        perr = np.round(np.sqrt((np.diag(pcov))),decimals=6)
        params = np.round(popt,decimals=6)
        fit = function(xscale,*popt)
        residuals = data - fit
        if plot:
            plotFitComparison(xscale,data,function,params,sigma)
            plotFitResiduals(xscale,residuals,sigma)
    except RuntimeError:
        print("Error - curve_fit failed")
        print(guess)
        plt.figure()
        plt.plot(xscale,data)
        plt.plot(xscale,function(xscale,*guess))
        params = []
        perr = []
        residuals = []
    return [params,perr,residuals]

def fitFlatTopGaussian(xscale,data,guess,sigma=None,bounds=None,plot=True,verbose=False):
    bounds = ([0,-np.inf,-np.inf,-np.inf,data.max()/2],[abs(xscale[-1]-xscale[0]),np.inf,np.inf,np.inf,np.inf])
    function = flatGaussian
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot,bounds=bounds)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
        print('Mean = {} +/- {} MHz, StDev = {} +/- {} MHz'.format(params[1],error[1],params[0],error[0]))
    return [params,error,residuals]

def fitFlatTopVoigt(xscale,data,guess,sigma=None,bounds=None,plot=True,verbose=False):
    bounds = ([0,0,-np.inf,-np.inf,-np.inf,data.max()/2],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
    function = flatVoigt
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot,bounds=bounds)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
    return [params,error,residuals]

def fitVoigtLinBG(xscale,data,guess,sigma=None,bounds=None,plot=True,verbose=False):
    bounds = ([0,0,-np.inf,-np.inf,-np.inf,-np.inf],[abs(xscale).max(),abs(xscale).max(),np.inf,np.inf,np.inf,np.inf])
    function = voigtLinBG
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot,bounds=bounds)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
    return [params,error,residuals]

def fitVoigt(xscale,data,guess,sigma=None,bounds=None,plot=True,verbose=False):
    bounds = ([0,0,-np.inf,-np.inf,-np.inf],[abs(xscale).max(),abs(xscale).max(),np.inf,np.inf,np.inf])
    function = voigt
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot,bounds=bounds)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
    return [params,error,residuals]



def fitInvertedLine(x,y,guess,xsigma,plot,verbose):
    # NOTE: inputs are all NOT inverted, the inversion is done in this function
    #y = mx + b, so y/m - b/m = x.
    #If we identify y'=x, x'=y, m'=1/m, b'=-b/m, we get y'=m'x'+b'
    #Here, y' = calibrated freq, and x' = means
    inv_x = y
    inv_y = x
    inv_ysigma = xsigma
    inv_guess = [1/guess[0],-guess[1]/guess[0]]
    params_inv,errors_inv,resid_inv = fitLine(inv_x,inv_y,inv_guess,sigma=inv_ysigma,plot=False,verbose=False)
    #Need to invert everything...m'=1/m, b'=-b/m
    slope = 1/params_inv[0]
    slope_err = errors_inv[0]/params_inv[0]*slope
    intercept = -params_inv[1]/params_inv[0]
    intercept_err = intercept * np.sqrt((errors_inv[1]/params_inv[1])**2 + (errors_inv[0]/params_inv[0])**2)
    params = [slope,intercept]
    error = [slope_err,intercept_err]
    fit = line(x,*params)
    residuals = y - fit
    if plot:
        plotFitComparison(x,y,line,params,None)
        plotFitResiduals(x,residuals,None)
    if verbose:
        print('Slope = {} +/- {}'.format(params[0],error[0]))
        print('Yintercept = {} +/- {}'.format(params[1],error[1]))
    return [params,error,residuals]

def fitLine0(xscale, data,guess=1,sigma=None,plot=True,verbose=False):
    function = line0
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot)
    if verbose:
        print('Slope = {} +/- {}'.format(params[0],error[0]))
    return [params,error,residuals]

def fitLine(xscale, data,guess=[1,0],sigma=None,plot=True,verbose=False):
    function = line
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot)
    if verbose:
        print('Slope = {} +/- {}'.format(params[0],error[0]))
        print('Yintercept = {} +/- {}'.format(params[1],error[1]))
    return [params,error,residuals]

def fitGaussian(xscale, data,guess=[100,0,0.16,0],sigma=None,plot=True,verbose=False):
    function = gaussian
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
        print('Mean = {} +/- {} MHz, StDev = {} +/- {} MHz'.format(params[1],error[1],params[0],error[0]))
    return [params,error,residuals]

def fitGaussianLinBG(xscale, data,guess=[100,0,0.16,0,0],sigma=None,plot=True,verbose=False):
    function = gaussianLinBG
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
        print('Mean = {} +/- {} MHz, StDev = {} +/- {} MHz'.format(params[1],error[1],params[0],error[0]))
    return [params,error,residuals]


def fit3Gaussians(xscale, data,mean_guess,stdev_guess,norm_guess,offset_guess=[0],sigma=None,plot=True,verbose=False):
    guess = list(stdev_guess) + list(mean_guess) + list(norm_guess)+list(offset_guess)
    if verbose:
        print(guess)
    function = threeGaussians
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
        for i in range(3):
            peaknum = i+1
            mean = params[3+i]
            mean_err = error[3+i]
            stdev = params[i]
            stdev_err = error[i]
            print('Mean {} = {} +/- {} MHz, StDev {} = {} +/- {} MHz'.format(peaknum,mean,mean_err,peaknum,stdev,stdev_err))
    return [params,error,residuals]

def fit2Gaussians(xscale, data,mean_guess,stdev_guess,norm_guess,offset_guess=[0],sigma=None,plot=True,verbose=False):
    guess = list(stdev_guess) + list(mean_guess) + list(norm_guess)+list(offset_guess)
    if verbose:
        print(guess)
    function = twoGaussians
    params,error,residuals = fitFunction(xscale,data,function,guess,sigma,plot)
    if verbose:
        print('\n')
        print('Fit error = ',error)
        print('FIT PARAMS = ',params)
        for i in range(2):
            peaknum = i+1
            mean = params[2+i]
            mean_err = error[2+i]
            stdev = params[i]
            stdev_err = error[i]
            print('Mean {} = {} +/- {} MHz, StDev {} = {} +/- {} MHz'.format(peaknum,mean,mean_err,peaknum,stdev,stdev_err))
    return [params,error,residuals]


###############################################################################
'''Generating Guesses for Fits'''
###############################################################################

def genGuessGaussian(xscale,data):
    peak_indices,properties = find_peaks(data, height =np.amax(data))
    peak_locs = xscale[peak_indices]
    peak_heights = properties['peak_heights']
    width_info = peak_widths(data,peak_indices)
    width_indices = width_info[0]
    deltax = abs(xscale[-1]-xscale[0])/len(xscale)
    widths = width_indices*deltax
    return [peak_locs,widths,peak_heights]

def genGuess3Gaussians(xscale, data):
    peak_indices,properties = find_peaks(data, height =np.amax(data)/5,distance=len(data)/6)
    peak_locs = xscale[peak_indices]
    peak_heights = properties['peak_heights']
    width_info = peak_widths(data,peak_indices)
    width_indices = width_info[0]
    deltax = abs(xscale[-1]-xscale[0])/len(xscale)
    widths = width_indices*deltax
    return [peak_locs,widths,peak_heights]


###############################################################################
'''Plotting'''
###############################################################################

def plotFitComparison(xscale,data,function,params,sigma,xlabel=None,ylabel=None):
    plt.figure()
    plt.title('Fit vs Data')
    if sigma is None:
        plt.plot(xscale,data,label='Data',marker='o')
    else:
        plt.errorbar(xscale,data,yerr=sigma,label='Data',marker='o',linestyle='None')
    plt.plot(xscale,function(xscale,*params),label='Fit')
    return

def plotFitResiduals(xscale,residuals,sigma):
    plt.figure()
    plt.title('Fit Residuals')
    if sigma is None:
        plt.plot(xscale,residuals,'o',linestyle='None')
    else:
        plt.errorbar(xscale,residuals,yerr=sigma,marker='o',linestyle='None')
    plt.plot(xscale,np.zeros(len(xscale)),linestyle='--')
    return


###############################################################################
'''f(x) = x Functions'''
###############################################################################
# Note these only work with numpy array inputs for x

def line(x,m,b):
    value = m*x+b
    return value

def line0(x,m):
    value = line(x,m,0)
    return value

def gaussian(x,a,b,n,c): # n exp(-(x-b)^2/(2a^2)) + c
    value= n*np.exp(-(x-b)**2/(2*a**2))+c
    return value

def gaussianLinBG(x,a,b,n,m,c): # n exp(-(x-b)^2/(2a^2)) + m*x+c
    value= n*np.exp(-(x-b)**2/(2*a**2))+line(x,m,c)
    return value

def flatTopFunction(x,function,func_params,threshold): # piece wise function. f(x)=f(x) when f(x)<s, otherwise f(x)=s
    value = function(x,*func_params)
    value[value>threshold]=threshold
    return value

def flatGaussian(x,a,b,n,c,threshold):
    params = [a,b,n,c]
    return flatTopFunction(x,gaussian,params,threshold)

def flatVoigt(x,lor,sig,mean,n,c,threshold):
    params = [lor,sig,mean,n,c]
    return flatTopFunction(x,voigt,params,threshold)

def voigt(x,lor,sig,mean,n,c): #lor = hwhm, sigma = gaussian stdev
    z= ((x-mean) + 1j*lor)/(sig*np.sqrt(2))
    value = n*np.real(wofz(z))/(sig*np.sqrt(2*np.pi))+c
    return value

def voigtLinBG(x,lor,sig,mean,n,m,b):
    value = n*np.real(wofz(((x-mean)+1j*lor)/(sig*np.sqrt(2))))/(sig*np.sqrt(2*np.pi))+line(x,m,b)
    return value

def lorentzian(x, gamma):
    return gamma / np.pi / (x**2 + gamma**2)

def twoGaussians(x,a1,a2,b1,b2,n1,n2,c=0):
    params1 = [a1,b1,n1,c]
    params2 = [a2,b2,n2,c]
    total = gaussian(x,*params1)+gaussian(x,*params2)
    return total

def threeGaussians(x,a1,a2,a3,b1,b2,b3,n1,n2,n3,c=0):
    params1 = [a1,b1,n1,c]
    params2 = [a2,b2,n2,c]
    params3 = [a3,b3,n3,c]
    total = gaussian(x,*params1)+gaussian(x,*params2)+gaussian(x,*params3)
    return total
