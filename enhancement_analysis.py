
import matplotlib.pyplot as plt
import pathlib
from scipy.optimize import curve_fit
import ipywidgets as widgets
from IPython.display import display
from traitlets import traitlets
import numpy as np
import statistics as stat
import scipy.signal as sig
from scipy.signal import savgol_filter,find_peaks,peak_widths
import csv
from math import ceil,floor
from fit_functions import *
from low_level import *

'''Global variables record indexing of parameter array for each trace'''
dt_INDEX = 0
start_INDEX = 1
nsample_INDEX = 2
trigtime_INDEX = 3
blocked_INDEX = 4
param_LENGTH = 5


###############################################################################
'''Processing Optical Depths'''
###############################################################################

'''Iterate through a series of files and calculate ODs and parameters from both
    channels. Returns an array of ODs and parameters.'''
def calculateSeriesOD(folder,start_num,stop_num,skips=[], ch=3, ABAB = False, ABBA = False, BAAB=False):
    stop_num+=1
    Ch1_ODs = []
    Ch2_ODs = []
    params = []
    progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    display(progress)
    delta = 1
    index = -1
    if ABAB:
    	delta = 2
    elif ABBA or BAAB:
        delta = 1
        index = 1
    for i in range(start_num,stop_num,delta):
        if index == 1: # (A)BBA
            index+=1
            if BAAB:
                continue
        elif index == 2 or 3: # A(B)BA or AB(B)A
            index+=1
            if ABBA:
                continue
        elif index == 4: # ABB(A)
            index = 1
            if BAAB:
                continue
        if i in skips:
            continue
        dataset = calculateSingleOD(folder,i,ch)
        _ODs = dataset[0]
        _params = dataset[-1]
        if len(_ODs)>1: #Both channels
            Ch1_ODs.append(_ODs[0])
            Ch2_ODs.append(_ODs[1])
        elif ch==1:
            Ch1_ODs.append(_ODs[0])
        elif ch==2:
            Ch2_ODs.append(_ODs[0])
        params.append(_params)
        progress.value = float(i-start_num+1)/(stop_num-start_num)
    progress.value=1
    if ch==1:
        dataset = [np.array(Ch1_ODs),np.array(params)]
    if ch==2:
        dataset = [np.array(Ch2_ODs),np.array(params)]
    else:
        dataset = [np.array(Ch1_ODs), np.array(Ch2_ODs),np.array(params)]
    return dataset

'''Integrate a series of ODs'''
def integrateODSeries(ODs,parameters,start_stop=[0,6],fignum=1):
    num_traces = len(ODs)
    integrated_ODs = np.zeros(num_traces)
    #progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    #display(progress)
    for i in range(num_traces):
        time_ms = timeArray(parameters[i])
        integrated_ODs[i] = sliceIntegrate(ODs[i],time_ms,start_stop,fignum)
        #if i==0:
        #    num_traces+=1
        #progress.value = float(i)/(num_traces-1)
    #progress.value=1
    return integrated_ODs



'''Returns integrated ODs and array of timestamps. Useful for chirp analysis'''
def processData(dataset, start_stop=[0,6]):
    YbOH_fig = 1
    plt.figure(YbOH_fig)
    plt.title('YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    Yb_fig = 2
    plt.figure(Yb_fig)
    plt.title('Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    params = dataset[-1]
    if len(dataset) == 3:
        YbOH_ODs = dataset[0]
        Yb_ODs = dataset[1]
        Skip_Yb=False
    else:
        YbOH_ODs = dataset[0]
        Skip_Yb = True

    YbOH_int_ODs = integrateODSeries(YbOH_ODs,params,start_stop,fignum=YbOH_fig)
    times = extractTimestamps(params)
    sorted_YbOH_int = sortData(times,YbOH_int_ODs)
    sorted_times = sorted(times)
    if Skip_Yb:
        results = [np.array(sorted_YbOH_int), np.array(sorted_times)]
    else:
        Yb_int_ODs = integrateODSeries(Yb_ODs,params,start_stop,fignum=Yb_fig)
        sorted_Yb_int = sortData(times,Yb_int_ODs)
        results = [np.array(sorted_YbOH_int), np.array(sorted_Yb_int), np.array(sorted_times)]
    return results

def processChirpWithBounds(dataset,freq_start,freq_stop, start_stop=[0,6]):
    if len(dataset)==3:
        YbOH, Yb, time = processScan(dataset,start_stop)
    else:
        YbOH, time = processScan(dataset,start_stop)
        Yb = np.zeros(len(time))
    deltat = time[-1]
    print('Chirp took {} seconds'.format(deltat))
    lightspeed = 29979.2458 #cm/us
    start = freq_start*lightspeed - 539300000
    stop = freq_stop*lightspeed - 539300000
    print('Start = {} MHz'.format(start))
    print('Stop = {} MHz'.format(stop))
    speed = (stop-start)/deltat
    print('Chirp Speed = {}'.format(speed))
    freq = []
    for point in time:
        value = start + point*speed
        freq.append(value)

    plt.figure(figsize=(7.5,5))
    plt.title('YbOH Enhancement Frequency Chirp')
    plt.xlabel('556 nm Frequency (5393xxxxx MHz)')
    plt.ylabel('Integrated YbOH OD')
    plt.plot(freq,YbOH)

    plt.figure(figsize=(7.5,5))
    plt.title('Yb Enhancement Frequency Chirp')
    plt.xlabel('556 nm Frequency (5393xxxxx MHz)')
    plt.ylabel('Integrated Yb OD')
    plt.plot(freq,Yb)
    return [freq,YbOH, Yb]


def processIVData(blocked_ODs, unblocked_ODs,IV,start_stop_YbOH_blocked=[0,3],start_stop_YbOH_unblocked=[0,3],start_stop_Yb_blocked=[0,3],start_stop_Yb_unblocked=[0,3]):
    blocked_int_dataset,unblocked_int_dataset = process_BUB(blocked_ODs, unblocked_ODs,start_stop_YbOH_blocked,start_stop_YbOH_unblocked,start_stop_Yb_blocked,start_stop_Yb_unblocked)
    sorted_YbOH_blocked = sortData(IV,blocked_int_dataset[0])
    sorted_Yb_blocked = sortData(IV,blocked_int_dataset[1])
    sorted_YbOH_unblocked = sortData(IV,unblocked_int_dataset[0])
    sorted_Yb_unblocked = sortData(IV,unblocked_int_dataset[1])
    sorted_IV = sorted(IV)
    YbOH_enhancement = calcEnhancement(sorted_YbOH_blocked,sorted_YbOH_unblocked)
    Yb_enhancement = calcEnhancement(sorted_Yb_blocked,sorted_Yb_unblocked)
    YbOH_int_data = [sorted_YbOH_blocked,sorted_YbOH_unblocked]
    Yb_int_data = [sorted_Yb_blocked,sorted_Yb_unblocked]
    return [YbOH_enhancement, Yb_enhancement, sorted_IV,YbOH_int_data,Yb_int_data]

def processTimeData(blocked_ODs, unblocked_ODs,start_stop_YbOH_blocked=[0,3],start_stop_YbOH_unblocked=[0,3],start_stop_Yb_blocked=[0,3],start_stop_Yb_unblocked=[0,3]):
    blocked_int_dataset,unblocked_int_dataset = processBUb(blocked_ODs, unblocked_ODs,start_stop_YbOH_blocked,start_stop_YbOH_unblocked,start_stop_Yb_blocked,start_stop_Yb_unblocked)
    blocked_params = blocked_int_dataset[-1]
    unblocked_params = unblocked_int_dataset[-1]
    blocked_times,unblocked_times = extractTimestamps(blocked_params,unblocked_params)
    sorted_YbOH_blocked = sortData(blocked_times,blocked_int_dataset[0])
    sorted_Yb_blocked = sortData(blocked_times,blocked_int_dataset[1])
    sorted_YbOH_unblocked = sortData(unblocked_times,unblocked_int_dataset[0])
    sorted_Yb_unblocked = sortData(unblocked_times,unblocked_int_dataset[1])
    sorted_blocked_times = sorted(blocked_times)
    sorted_unblocked_times = sorted(unblocked_times)
    avg_times = (np.array(sorted_blocked_times)+np.array(sorted_unblocked_times))/2
    YbOH_enhancement = calcEnhancement(sorted_YbOH_blocked,sorted_YbOH_unblocked)
    Yb_enhancement = calcEnhancement(sorted_Yb_blocked,sorted_Yb_unblocked)
    time_info = [sorted_blocked_times,sorted_unblocked_times,avg_times]
    YbOH_int_data = [sorted_YbOH_blocked,sorted_YbOH_unblocked]
    Yb_int_data = [sorted_Yb_blocked,sorted_Yb_unblocked]
    return [YbOH_enhancement, Yb_enhancement, time_info,YbOH_int_data,Yb_int_data]

def processTimeDataSingle(dataset,start_stop=[0,3]):
    fig = 1

    plt.figure(fig)
    plt.title('YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    ODs = dataset[0]
    params = dataset[1]

    YbOH_int_ODs = integrateODSeries(ODs,params,start_stop,fignum=fig)
    times = extractTimestampsFixed(params)
    sorted_int_ODs = sortData(times,YbOH_int_ODs)
    sorted_times = sorted(times)
    return [sorted_int_ODs,sorted_times]




#########################################################################################################################################################


'''Analayis Functions'''

#def processBackAndForth(Yb_spectra,YbOH_spectra,times,range):
    #Divide files into individual scans
        #Detect Yb peaks
        #Divide spectra along half-way line
        #Map time array onto frequency range
    #Recenter scans on Yb peak
    #

def calib556_3peaks(Yb_spectra,method,plot=False,verbose=False):
    #Assuming data was taken at even intervals. Assuming data starts at low freq and ends at high, and contains 3 peaks corresponding to 176Yb, 174Yb, and 172Yb.
    #First, generate a scale in number of samples
    x = np.arange(len(Yb_spectra))
    #Now, we want to set the middle Yb peak to be our zero.
    #To do this, we will first fit 3 gaussians to the distribution, then find the mean of the middle Gaussian, and use that to offset our x scale.
    x_shift,xmeans,xerrs = shift3Gaussians(x,Yb_spectra)
    #Now, fit the peak locations vs actual frequency to a line, constrain y intercept=0
    Yb_176 = -954.832 #MHz detuning from 174Yb line
    Yb_172 = 1000.02 #MHz detuning from 174Yb line
    isoshifts = np.array([Yb_176,0,Yb_172])
    slope_guess = (isoshifts[-1]-isoshifts[0])/(xmeans[-1]-xmeans[0])
    if method == 1: #Fit y=shifts, x=means to a line y=mx
        params_l,errors_l,resid_l = fitLine0(xmeans,isoshifts,slope_guess,plot=plot,verbose=verbose)
        slope = params_l[0]
        intercept = 0
    elif method == 2: #Fit y=shifts, x=means to a line y=mx+b
        params_l,errors_l,resid_l = fitLine(xmeans,isoshifts,[slope_guess,0],plot=plot,verbose=verbose)
        slope = params_l[0]
        intercept = params_l[1]
    elif method == 3: #Fit y=means, x=shifts, yerr=mean error to a line y=mx+b, then invert. Do this to incorporate error in mean position
        params_l,errors_l,resid_l = fitInvertedLine(xmeans,isoshifts,[slope_guess,0],xerrs,plot,verbose)
        slope = params_l[0]
        intercept = params_l[1]
    #Calculate calibrated frequencies
    freq_calib = x_shift * slope + intercept
    means_calib = xmeans * slope + intercept
    # We actually don't want the fit residuals, we want the detuning deviations!
    detuning_176 = means_calib[0] - means_calib[1]
    detuning_172 = means_calib[2] - means_calib[1]
    resid_detuning = [detuning_176-Yb_176,detuning_172-Yb_172]
    return [freq_calib,means_calib,resid_detuning]

def calib556_array(Yb_array,method,compare=[],plot=False,verbose=False,fig=1000):
    freq_calib = []
    means_calib = []
    resid_detuning_calib = []
    xresid_flat = []

    plt.figure(fig)
    plt.title('Fit Detuning Residuals')
    xticks = [0,1]
    plt.plot(xticks,np.zeros(len(xticks)),linestyle='--',color='black')
    xlabel = ['176Yb','172Yb']
    plt.xticks(xticks,xlabel)
    plt.ylabel('Deviation from Actual Shift (MHz)')
    plt.xlabel('Yb Isotope Detuning from 174Yb')

    for Yb in Yb_array:
        freq,means,detunings = calib556_3peaks(Yb,method,plot,verbose)
        freq_calib.append(freq)
        means_calib.append(means)
        resid_detuning_calib.append(detunings)
        xresid_flat.extend(xticks)
    resid_calib_flat = np.array(resid_detuning_calib).flatten()
    xticks_flat = np.array(xresid_flat)

    plt.figure(fig)
    plt.plot(xticks_flat,resid_calib_flat,marker='o',label='Calibrated Method {}'.format(method),linestyle='None')
    plt.legend(loc='best')
    if len(compare):
        i = 0
        resid_detuning_compare = []
        for _Yb,_calib,_compare,_means_calib in zip(Yb_array,freq_calib,compare,means_calib):
            if abs(_compare[int(len(_compare)/2)]) > 1000:
                _compare_shift,_compare_means,_compare_err = shift3Gaussians(_compare,_Yb)
                _compare_176 = (_compare_means[0] - _compare_means[1]) - (-954.328)
                _compare_172 = _compare_means[2] -_compare_means[1] - (1000.02)
                resid_detuning_compare.append(np.array([_compare_176,_compare_172]))
            if plot:
                plt.figure()
                plt.plot(_calib,_Yb, label='Calibrated')
                plt.plot(_compare_shift,_Yb,label='Uncalibrated')
                plt.legend(loc='best')
            if verbose:
                title = 'Scan {}\n'.format(i)
                s176 = '176Yb\nActual = {} MHz, Wavemeter = {} MHz, Cal = {} MHz\n'.format(-954.328,_compare_means[0],_means_calib[0])
                s174 = '174Yb\nActual = {} MHz, Wavemeter = {} MHz, Cal = {} MHz\n'.format(0,_compare_means[1],_means_calib[1])
                s172 = '172Yb\nActual = {} MHz, Wavemeter = {} MHz, Cal = {} MHz\n'.format(1000.02,_compare_means[2],_means_calib[2])
                print(title,s176,s174,s172)
            i+=1
        resid_compare_flat = np.array(resid_detuning_compare).flatten()
        plt.figure(fig)
        plt.plot(xticks_flat,resid_compare_flat,marker='o',label='Wavemeter',linestyle='None')
        plt.legend(loc='best')
    return [np.array(freq_calib),np.array(resid_detuning_calib),[xticks_flat,resid_calib_flat]]


def averageSpectra(freq_array,spectra_array,bin_size=10):
    freq_array = np.array(freq_array)
    spectra_array = np.array(spectra_array)
    pos = freq_array[:][-1].max()
    neg = freq_array[:][0].min()
    num_bins = int(ceil((pos-neg)/bin_size))
    bins = np.arange(neg,pos,bin_size)
    _sum = np.zeros(len(bins))
    num_avg = np.zeros(len(bins))
    for i in range(len(bins)):
        for j in range(len(freq_array)):
            for k in range(len(freq_array[j])):
                if bins[i] <= freq_array[j][k] < bins[i] + bin_size:
                    _sum[i]+= spectra_array[j][k]
                    num_avg[i]+=1
    avg_spectra = _sum/num_avg
    return [bins,avg_spectra]


def processBackAndForth(Yb_spectra,YbOH_spectra,time_array,chirp_speed,num_peaks,initial_up):
    Yb = np.array(Yb_spectra)
    YbOH = np.array(YbOH_spectra)
    times = np.array(time_array)
    #Split data into individual chirps
    YbOH_split,Yb_split,times_split = splitData(YbOH,Yb,times,num_peaks,initial_up)

    #Use timing information and chirp speed to back out frequency info
    freq_split = []
    for _time in times_split:
        _freq = genFreqFromTime(_time,chirp_speed)
        freq_split.append(_freq)

    #Shift Yb fit to zero
    freq_shifted = []
    for _freq,_Yb in zip(freq_split,Yb_split):
        _shifted = shiftFlatTop(_freq,_Yb,method=2)
        freq_shifted.append(_shifted)

    return [YbOH_split,Yb_split,freq_shifted]

def genFreqFromTime(time,speed):
    #Speed and time should both be in seconds!
    n = len(time)
    freq = np.zeros(n)
    for i in range(1,n):
        deltaf = speed*(time[i] - time[i-1])
        freq[i] = deltaf+freq[i-1]
    return freq

def splitData(YbOH,Yb,times,num_peaks,initial_up):
    #Find peaks
    peak_indices, properties = find_peaks(Yb,height=Yb.max()/3,distance=(len(Yb)/(num_peaks*2)))
    if len(peak_indices) != num_peaks:
        print('Error: found {} peaks instead of {}'.format(len(peak_indices),num_peaks))
    Yb_split = []
    YbOH_split = []
    times_split = []
    print(len(Yb))
    for i in range(num_peaks):
        if i == 0: #Initial condition
            start = 0
            up = initial_up
        else: #start where the previous loop ended, switch up to down and vice versa
            start = end
            up = not up
        if i != num_peaks-1: #If not the last index
            halfway = int(round((peak_indices[i+1] - peak_indices[i])/2))
            end = peak_indices[i]+halfway
        else: #If last index, just go to the end
            end = None
        s = slice(start,end)
        #If the scan is downward, flip the spectra so we can compare everything equally
        if up:
            Yb_slice = Yb[s]
            YbOH_slice = YbOH[s]
            _time_slice = times[s]
        else:
            Yb_slice = np.flip(Yb[s])
            YbOH_slice = np.flip(YbOH[s])
            _time_slice = np.flip(times[s])
        Yb_split.append(Yb_slice)
        YbOH_split.append(YbOH_slice)
        time_slice = _time_slice - _time_slice[0]
        times_split.append(time_slice)
        print(len(Yb[s]))
    _sum = 0
    for thing in Yb_split:
        _sum+=len(thing)
    print(_sum)
    return [YbOH_split,Yb_split,times_split]

def identifyBUB(a1,a2,equal=False):
    _a1 = a1
    _a2 = a2
    if equal:
        delta = len(a1[0]) - len(a2[0])
        if delta > 0:
            _a1 = [a1[i][:-delta] for i in range(len(a1))]
        elif delta < 0:
            _a2 = [a2[i][:-delta] for i in range(len(a2))]
    if _a1[0].mean() > _a2[0].mean():
        unblocked = _a1
        blocked = _a2
    else:
        unblocked = _a2
        blocked = _a1
    return [blocked,unblocked]

def matchBUB(blocked,unblocked):
    b_match = []
    ub_match = []
    for _barray,_ubarray in zip(blocked,unblocked):
        _b_split = []
        _ub_split = []
        for _b,_ub in zip(_barray,_ubarray):
            delta = len(_ub) - len(_b)
            if delta>0:
                _ub = _ub[:-delta]
            elif delta<0:
                _b = _b[:delta]
            _b_split.append(_b)
            _ub_split.append(_ub)
        b_match.append(_b_split)
        ub_match.append(_ub_split)
    return [b_match,ub_match]

#################################################################################################################################
