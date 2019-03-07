
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

'''Global variables record indexing of parameter array for each trace'''
dt_INDEX = 0
start_INDEX = 1
nsample_INDEX = 2
trigtime_INDEX = 3
blocked_INDEX = 4
param_LENGTH = 5


############################################# Raw Data Functions #############################################

'''Read/write rows from/to CSV'''
def writeCSVrows(array_of_arrays,filepath):
    with open(filepath,'w+',newline='') as f:
        w = csv.writer(f)
        for array in array_of_arrays:
            w.writerow(array)
    print('Done! Array written to', filepath)
    return

def readCSVrows(filepath):
    with open(filepath,'r') as f:
        r = csv.reader(f)
        rows = np.array([np.array([float(value) for value in row]) for row in r])
    print('Done! Array read from', filepath)
    return rows

'''Import raw traces from data columns in single cleverscope text file'''
def importRawData(filepath,usecols=(1,2)):
    #generate array from text file
    try:
        arr = np.genfromtxt(filepath,delimiter='', skip_header = 15, usecols=usecols)
        ch1_raw = arr.T[0] #Channel A
        ch2_raw = arr.T[1] #Channel B
        raw_data = [ch1_raw,ch2_raw]
    except ValueError:
        print('Issues with file located at: ',filepath)
        raw_data = [np.zeros(10000),np.zerps(10000)]
    return raw_data

'''Grab trace parameters from single cleverscope text file'''
def importCleverParams(filepath):
    with open(filepath, 'r') as f:
        lines=[]
        for i in range(15):
            lines.append(f.readline())
    params=np.zeros(param_LENGTH)
    for text in lines:
        if 'delta' in text:
            dt_ms = np.round(float(text.split('\t')[1].strip())*10**3,decimals=6)
            params[dt_INDEX]=dt_ms
        elif 'start' in text:
            start_ms = np.round(float(text.split('\t')[1].strip())*10**3,decimals = 6)
            params[start_INDEX]=start_ms
        elif 'nsample' in text:
            nsample = int(float(text.split('\t')[1].strip()))
            params[nsample_INDEX]=nsample
        elif 'TriggerTime' in text:
            time_sec = float(text.split('\t')[1].strip())*24*3600 #convert days to seconds
            params[trigtime_INDEX]=time_sec
    return params

'''Generate filepath for single text file named "spectra_num.txt"'''
def genPath(root_folder,num):
    num_str = str(num)
    name = 'spectra_'
    file =  name + num_str + '.txt'
    #path is given from current working directory
    filepath = pathlib.Path.cwd() / root_folder / file
    return filepath

'''From a single file location, obtain dataset with raw data and associated parameters'''
def file2dataset(root_folder,num,print_bool=False):
    filepath = genPath(root_folder,num)
    if print_bool:
        print(filepath)
    data = importRawData(filepath)
    parameters = importCleverParams(filepath)
    dataset = [data,parameters]
    return dataset

'''Convert many raw text files to data arrays.
    This code is written to handle data that alternates between blocked/unblocked.
    Initial = True/False refers to the first file being Blocked/Unblocked
    Alternate = True/False refers to if the order goes B Ub Ub B (True) or B Ub B Ub (False)
    Returns dataset with blocked and unblocked raw traces'''
def getRawDataset_BUB(folder_path,start_num,stop_num,print_bool,skips=[],initial=False,alternate=False):
    stop_num+=1
    blocked_YbOH_raw = [] #ch1
    blocked_Yb_raw = [] #ch2
    blocked_params = []
    unblocked_YbOH_raw = []
    unblocked_Yb_raw = []
    unblocked_params=[]

    blocked = initial
    switch_b_ub = True

    progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    display(progress)
    for i in range(start_num,stop_num):
        if i in skips:
            pass
        else:
            if blocked:
                data,params = file2dataset(folder_path,i,print_bool)
                params[blocked_INDEX]=blocked
                blocked_YbOH_raw.append(data[0])
                blocked_Yb_raw.append(data[1])
                blocked_params.append(params)
                #print('blocked ',i)
            elif not blocked:
                data,params = file2dataset(folder_path,i,print_bool)
                params[blocked_INDEX]=blocked
                unblocked_YbOH_raw.append(data[0])
                unblocked_Yb_raw.append(data[1])
                unblocked_params.append(params)
                #print('unblocked ',i)
            if alternate:
                if switch_b_ub:
                    blocked = not blocked
                    switch_b_ub = False
                else:
                    switch_b_ub = True
            else:
                blocked = not blocked
        progress.value = float(i-start_num+1)/(stop_num-start_num)
    blocked_dataset = [blocked_YbOH_raw, blocked_Yb_raw, blocked_params]
    unblocked_dataset = [unblocked_YbOH_raw, unblocked_Yb_raw, unblocked_params]
    return [blocked_dataset,unblocked_dataset]

# '''Widget wrapper for getRawDataset'''
# def wrapper_getRawDataset(w):
#         print('Generating raw data from ',folder_path_w.value)
#         #results = processData(folder_path_w.value,start_num_w.value,stop_num_w.value,start_setpnt_w.value,stop_setpnt_w.value,setpnt_int_w.value,print_w.value)
#         results = getRawDataset(folder_path_w.value,start_num_w.value,stop_num_w.value,print_w.value)
#         w.value = results
#         print('Done')



############################################# Data Processing Functions #############################################


'''Convert raw absorption data into optical depth'''
def raw2OD(raw_data,time_ms):
    trigger_index = np.searchsorted(time_ms,0)
    beforeYAG_index = np.searchsorted(time_ms,-0.1)
    #Calculate DC offset, convert signal to OD
    offset = raw_data[:beforeYAG_index].mean()
    #Smooth the data
    smoothed_data = smooth(raw_data,window=60)
    smoothed_data[smoothed_data<0] = 0.00001
    #Calculate OD, fix floating point errors
    OD = np.log(offset/smoothed_data)
    if OD[trigger_index]<0:
        offset = smoothed_data[trigger_index]
        OD = np.log(offset/smoothed_data)
    return OD

'''Function for smoothing data. Currently uses Savitzky-Golay filter, which fits a window
    onto a polynomial of some order, and then uses the polynomial to estimate the value'''
def smooth(data,window=5,poly_order=3):
    #window value must be odd
    if window % 2 == 0:
        window+=1
    smoothed_data = savgol_filter(data, window, poly_order)
    return smoothed_data

#Not used anymore. Only useful if there is linear drift
# def subtractBackground(raw,time):
#     end = len(cell_abs)
#     start = 5000
#     t = slice(start,end)
#     offset = -1.8486
#     num_avg = 900
#     time_ms = np.round(np.linspace(offset,dt*(end-1)+offset,end),decimals = 6)
#     b_guess = cell_abs[:num_avg].sum()/num_avg
#     m_guess = 0
#     abs_slice = cell_abs[t]
#     time_slice = time_ms[t]
#     params = fitLine(time_slice,abs_slice,guess=[m_guess,b_guess])
#     background = line(time_ms,*params)
#     cell_OD = np.log(background/cell_abs)
#     #cell_OD[cell_OD<0]=0
#     return cell_OD

'''Generate time array with time series information'''
def timeArray(parameters):
    dt = parameters[dt_INDEX]
    t0 = parameters[start_INDEX]
    npnts = parameters[nsample_INDEX]
    time_ms = np.round(np.linspace(t0,dt*(npnts-1)+t0,npnts),decimals = 6)
    return time_ms

'''Calculate the optical depth (OD) signal from a single text file.
    Returns a dataset, which consists of the optical dpeth and acquisition parameters'''
def calculateSingleOD(root_folder,num):
    raw_traces,params = file2dataset(root_folder,num)
    OD_ch1 = raw2OD(np.array(raw_traces[0]),timeArray(params))
    OD_ch2 = raw2OD(np.array(raw_traces[1]),timeArray(params))
    return [OD_ch1,OD_ch2,params]


'''Iterate through a series of files and calculate ODs and parameters from both
    channels. Returns an array of ODs and parameters.'''
def calculateSeriesODFromRaw(folder,start_num,stop_num,skips=[]):
    stop_num+=1
    Ch1_ODs = []
    Ch2_ODs = []
    params = []
    progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    display(progress)
    for i in range(start_num,stop_num):
        if i in skips:
            pass
        else:
            dataset = calculateSingleOD(folder,i)
            Ch1_ODs.append(dataset[0])
            Ch2_ODs.append(dataset[1])
            params.append(dataset[-1])
        progress.value = float(i-start_num+1)/(stop_num-start_num)
    dataset = [np.array(Ch1_ODs), np.array(Ch2_ODs),np.array(params)]
    return dataset

'''Iterate through a series of files and compiles ODs and parameters.
    Labels the whole series as blocked (True) or unblocked (False).
    Returns an array of ODs and parameters'''
def label_BUB(param_array,blocked_bool):
    for single_param in param_array:
        single_param[blocked_INDEX] = blocked_bool
    return [ODs,params]


'''This function is specifically written to work with getRawDataset_BUB().'''
def calculateODs_BUB(blocked_dataset,unblocked_dataset):

    time_length = len(blocked_dataset[0][0])
    num_traces = len(blocked_dataset[0])

    blocked_YbOH_OD = np.zeros((num_traces,time_length))
    unblocked_YbOH_OD = np.zeros((num_traces,time_length))
    blocked_Yb_OD = np.zeros((num_traces,time_length))
    unblocked_Yb_OD = np.zeros((num_traces,time_length))
    unblocked_params = []
    blocked_params = []

    plt.figure(1)
    plt.title('Blocked YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(2)
    plt.title('Unblocked YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(3)
    plt.title('Blocked Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(4)
    plt.title('Unblocked Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    display(progress)
    for dataset in [blocked_dataset,unblocked_dataset]:
        YbOH = dataset[0]
        Yb = dataset[1]
        params = dataset[2]
        for i in range(num_traces):
            time_ms = timeArray(params[i])
            YbOH_OD = raw2OD(YbOH[i],time_ms)
            Yb_OD = raw2OD(Yb[i],time_ms)
            blocked = params[i][4]
            if blocked:
                plt.figure(1)
                plt.plot(time_ms,YbOH_OD)
                plt.figure(3)
                plt.plot(time_ms,Yb_OD)
                blocked_params.append(params[i])
                for j in range(time_length):
                    blocked_YbOH_OD[i][j]=YbOH_OD[j]
                    blocked_Yb_OD[i][j]=Yb_OD[j]
            elif not blocked:
                plt.figure(2)
                plt.plot(time_ms,YbOH_OD)
                plt.figure(4)
                plt.plot(time_ms,Yb_OD)
                unblocked_params.append(params[i])
                for j in range(time_length):
                    unblocked_YbOH_OD[i][j]=YbOH_OD[j]
                    unblocked_Yb_OD[i][j]=Yb_OD[j]
            progress.value = float(i)/(num_traces-1)
    blocked_ODs = [blocked_YbOH_OD, blocked_Yb_OD,blocked_params]
    unblocked_ODs = [unblocked_YbOH_OD, unblocked_Yb_OD,unblocked_params]

    return [blocked_ODs, unblocked_ODs]

# '''Widget wrapper, not used'''
# def wrapper_calcODs(w):
#         print('Calculating ODs from ',folder_path_w.value)
#         #results = processData(folder_path_w.value,start_num_w.value,stop_num_w.value,start_setpnt_w.value,stop_setpnt_w.value,setpnt_int_w.value,print_w.value)
#         results = calculateODs(*rawdata_lb.value)
#         w.value = results
#         print('Done')
#         return

'''Extracts trigger timestamps from parameter arrays. Times are in seconds relative to the initial file timestamp'''
def extractTimestamps(params):
    timestamps = []
    for single_params in params:
        if len(timestamps)==0:
            initial = single_params[trigtime_INDEX]
            timestamps.append(0)
        else:
            elapsed = single_params[trigtime_INDEX] - initial
            timestamps.append(elapsed)
    return np.array(timestamps)


'''Extracts trigger timestamps from blocked/unblocked parameter arrays, and averages the two times together'''
def extractTimestamps_BUB(blocked_params,unblocked_params):
    blocked_timestamps = []
    unblocked_timestamps = []
    if blocked_params[0][trigtime_INDEX] < unblocked_params[0][trigtime_INDEX]:
        blocked_first = True
    else:
        blocked_first = False
    for i in range(len(blocked_params)):
        if blocked_first:
            if len(blocked_timestamps)==0:
                initial = blocked_params[i][trigtime_INDEX]
                blocked_timestamps.append(0)
                elapsed_unblocked = unblocked_params[i][trigtime_INDEX] - initial
                unblocked_timestamps.append(elapsed_unblocked)
            else:
                elapsed_blocked = blocked_params[i][trigtime_INDEX] - initial
                elapsed_unblocked = unblocked_params[i][trigtime_INDEX] - initial
                blocked_timestamps.append(elapsed_blocked)
                unblocked_timestamps.append(elapsed_unblocked)
        else:
            if len(unblocked_timestamps)==0:
                initial = unblocked_params[i][trigtime_INDEX]
                unblocked_timestamps.append(0)
                elapsed_blocked = blocked_params[i][trigtime_INDEX] - initial
                blocked_timestamps.append(elapsed_blocked)
            else:
                elapsed_blocked = blocked_params[i][trigtime_INDEX] - initial
                elapsed_unblocked = unblocked_params[i][trigtime_INDEX] - initial
                blocked_timestamps.append(elapsed_blocked)
                unblocked_timestamps.append(elapsed_unblocked)
    return [blocked_timestamps,unblocked_timestamps]

'''Integrate time slice of the OD from start to stop.
    start_stop = [start,stop] in ms'''
def sliceIntegrate(OD,time_ms,start_stop,fig_num=1):
    start_i = np.searchsorted(time_ms,start_stop[0])
    stop_i = np.searchsorted(time_ms,start_stop[1])
    t = slice(start_i,stop_i)
    dt = np.round(time_ms[1]-time_ms[0],decimals=6)
    integrated = np.round(OD[t].sum()*dt,decimals=6)
    plt.figure(fig_num)
    plt.plot(time_ms[t],OD[t])
    return integrated

'''Integrate a series of ODs'''
def integrateODSeries(ODs,parameters,start_stop=[0,3],fignum=1):
    num_traces = len(ODs)
    integrated_ODs = np.zeros(num_traces)
    progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    display(progress)
    for i in range(num_traces):
        time_ms = timeArray(parameters[i])
        integrated_ODs[i] = sliceIntegrate(ODs[i],time_ms,start_stop,fignum)
        if i==0:
            num_traces+=1
        progress.value = float(i)/(num_traces-1)
    return integrated_ODs


''''''
def processScan(dataset, start_stop=[0,3]):
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

    YbOH_ODs = dataset[0]
    Yb_ODs = dataset[1]
    params = dataset[-1]

    YbOH_int_ODs = integrateODSeries(YbOH_ODs,params,start_stop,fignum=YbOH_fig)
    Yb_int_ODs = integrateODSeries(Yb_ODs,params,start_stop,fignum=Yb_fig)
    times = extractTimestamps(params)
    sorted_YbOH_int = sortData(times,YbOH_int_ODs)
    sorted_Yb_int = sortData(times,Yb_int_ODs)
    #sorted_ODs = sortData(times,ODs)
    sorted_times = sorted(times)
    return [sorted_YbOH_int, sorted_Yb_int, sorted_times]

def processChirp(dataset,freq_start,freq_stop, start_stop=[0,6]):
    YbOH, Yb, time = processScan(dataset,start_stop)
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


def process_BUB(blocked_ODs, unblocked_ODs,start_stop_YbOH_blocked=[0,3],start_stop_YbOH_unblocked=[0,3],start_stop_Yb_blocked=[0,3],start_stop_Yb_unblocked=[0,3]):

    YbOH_B_fig = 5
    YbOH_Ub_fig = 6
    Yb_B_fig = 7
    Yb_Ub_fig = 8

    plt.figure(YbOH_B_fig)
    plt.title('Blocked YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(YbOH_Ub_fig)
    plt.title('Unblocked YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(Yb_B_fig)
    plt.title('Blocked Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(Yb_Ub_fig)
    plt.title('Unblocked Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    for dataset in [blocked_ODs, unblocked_ODs]:
        YbOH_OD = dataset[0]
        Yb_OD = dataset[1]
        params = dataset[2]
        blocked = params[0][4]
        if blocked:
            blocked_YbOH_int = integrateODSeries(YbOH_OD,params,start_stop_YbOH_blocked,fignum=YbOH_B_fig)
            blocked_Yb_int = integrateODSeries(Yb_OD,params,start_stop_Yb_blocked,fignum=Yb_B_fig)
        elif not blocked:
            unblocked_YbOH_int = integrateODSeries(YbOH_OD,params,start_stop_YbOH_unblocked,fignum=YbOH_Ub_fig)
            unblocked_Yb_int = integrateODSeries(Yb_OD,params,start_stop_Yb_unblocked,fignum=Yb_Ub_fig)
    blocked_params = blocked_ODs[2]
    unblocked_params = unblocked_ODs[2]

    blocked_int_dataset = [blocked_YbOH_int,blocked_Yb_int,blocked_params]
    unblocked_int_dataset = [unblocked_YbOH_int,unblocked_Yb_int,unblocked_params]

    return [blocked_int_dataset,unblocked_int_dataset]

def sortData(indep_var,depend_var):
    #print(len(indep_var))
    #print(len(depend_var))
    if len(indep_var) != len(depend_var):
        print('Number of independent and dependent variables do not match')
    else:
        if not isinstance(indep_var,list):
            indep_var = list(indep_var)
        sorted_DV = [DV for IV,DV in sorted(zip(indep_var,depend_var))]
    return sorted_DV

def calcEnhancement(blocked_int,unblocked_int):
    num_traces = len(blocked_int)
    enhancement = np.zeros(num_traces)
    for i in range(num_traces):
        if blocked_int[i]<0.0001:
            enhancement[i]=0
        else:
            enhancement[i] = unblocked_int[i]/blocked_int[i]
    return enhancement


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

def OLD_integrateODs(blocked_ODs, unblocked_ODs,start_stop_YbOH_blocked=[0,3],start_stop_YbOH_unblocked=[0,3],start_stop_Yb_blocked=[0,3],start_stop_Yb_unblocked=[0,3],indep_var=False):

    if not indep_var:
        indep_var = [i for i in range(len(blocked_ODs[0]))]

    num_traces = len(indep_var)

    blocked_YbOH_int = np.zeros(num_traces)
    blocked_Yb_int = np.zeros(num_traces)
    unblocked_YbOH_int = np.zeros(num_traces)
    unblocked_Yb_int = np.zeros(num_traces)
    YbOH_enhancement = np.zeros(num_traces)
    Yb_enhancement = np.zeros(num_traces)

    plt.figure(5)
    plt.title('Blocked YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(6)
    plt.title('Unblocked YbOH OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(7)
    plt.title('Blocked Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    plt.figure(8)
    plt.title('Unblocked Yb OD')
    plt.xlabel('Time (ms)')
    plt.ylabel('OD')

    progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
    display(progress)
    for dataset in [blocked_ODs, unblocked_ODs]:
        YbOH_OD = dataset[0]
        Yb_OD = dataset[1]
        params = dataset[2]
        for i in range(num_traces):
            time_ms = timeArray(params[i])
            blocked = params[i][4]
            if blocked:
                blocked_YbOH_int[i] = sliceIntegrate(YbOH_OD[i],time_ms,start_stop_YbOH_blocked,5)
                blocked_Yb_int[i] = sliceIntegrate(Yb_OD[i],time_ms,start_stop_Yb_blocked,7)
            elif not blocked:
                unblocked_YbOH_int[i] = sliceIntegrate(YbOH_OD[i],time_ms,start_stop_YbOH_unblocked,6)
                unblocked_Yb_int[i] = sliceIntegrate(Yb_OD[i],time_ms,start_stop_Yb_unblocked[0],8)
            progress.value = float(i)/(num_traces-1)
    unblocked_YbOH_int_sorted = [OD for power,OD in sorted(zip(indep_var,unblocked_YbOH_int))]
    unblocked_Yb_int_sorted = [OD for power,OD in sorted(zip(indep_var,unblocked_Yb_int))]
    blocked_YbOH_int_sorted = [OD for power,OD in sorted(zip(indep_var,blocked_YbOH_int))]
    blocked_Yb_int_sorted = [OD for power,OD in sorted(zip(indep_var,blocked_Yb_int))]

    YbOH_int_ODs = [unblocked_YbOH_int_sorted,blocked_YbOH_int_sorted]
    Yb_int_ODs = [unblocked_Yb_int_sorted,blocked_Yb_int_sorted]
    for i in range(num_traces):
        if blocked_YbOH_int_sorted[i]<0.0001:
            YbOH_enhancement[i]=0
        elif blocked_Yb_int_sorted[i]<0.001:
            Yb_enhancement[i]=0
        else:
            YbOH_enhancement[i] = unblocked_YbOH_int_sorted[i]/blocked_YbOH_int_sorted[i]
            Yb_enhancement[i] = unblocked_Yb_int_sorted[i]/blocked_Yb_int_sorted[i]

    indep_var_sorted = sorted(indep_var)
    return [YbOH_enhancement, Yb_enhancement, indep_var_sorted,YbOH_int_ODs,Yb_int_ODs]


def processAll(folder_path,start_num,stop_num,skips=[],initial_blocked=False,alternate=False,start_stop_YbOH_blocked=[0,2],start_stop_YbOH_unblocked=[0,3],start_stop_Yb_blocked=[0,2],start_stop_Yb_unblocked=[0,3],indep_var=False,print_bool=False):
    print('Getting raw data...')
    raw = getRawDataset(folder_path,start_num,stop_num,print_bool,skips,initial_blocked,alternate)
    print('Done!')
    print('Calculating ODs...')
    ODs = calculateODs(*raw)
    print('Done!')
    print('Integrating...')
    final_data = integrateODSeries(*ODs,start_stop_YbOH_blocked,start_stop_YbOH_unblocked,start_stop_Yb_blocked,start_stop_Yb_unblocked,indep_var)
    print('Done!')
    return final_data


def getODsfromBUB(folder_path,start_num,stop_num,skips=[],initial_blocked=False,alternate=False,print_bool=False):
    print('Getting raw data...')
    raw= getRawDataset_BUB(folder_path,start_num,stop_num,print_bool,skips,initial_blocked,alternate)
    print('Done!')
    print('Calculating ODs...')
    blocked_OD_dataset,unblocked_OD_dataset = calculateODs_BUB(*raw)
    print('Done!')
    return [blocked_OD_dataset,unblocked_OD_dataset]




#########################################################################################################################################################

'''Analayis Functions'''

def backAndForth(Yb_spectra,YbOH_spectra):
    

def calib556_3peaks(Yb_spectra,method,plot=False,verbose=False):
    #Assuming data was taken at even intervals. Assuming data starts at low freq and ends at high, and contains 3 peaks corresponding to 176Yb, 174Yb, and 172Yb.
    #First, generate a scale in number of samples
    x = np.arange(len(Yb_spectra))
    #Now, we want to set the middle Yb peak to be our zero.
    #To do this, we will first fit 3 gaussians to the distribution, then find the mean of the middle Gaussian, and use that to offset our x scale.
    x_shift,xmeans,xerrs = shift3Gaussians_2Zero(x,Yb_spectra)
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
                _compare_shift,_compare_means,_compare_err = shift3Gaussians_2Zero(_compare,_Yb)
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

def shift3Gaussians_2Zero(xscale,data):
    mean_guess,stdev_guess,norm_guess = genGuess3Gaussians(xscale,data)
    params_3g,err_3g,resid_3g = fit3Gaussians(xscale, data, mean_guess, stdev_guess, norm_guess,offset_guess=[data.min()],plot=False,verbose=False)
    means = params_3g[3:6]
    means_err = err_3g[3:6]
    #Offset so that the middle peak is at 0
    x_shift = xscale - means[1]
    means_shifted = np.array(means)-means[1]
    return [x_shift,means_shifted,means_err]


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


#################################################################################################################################



'''Widgets'''

style = {'description_width': 'initial'}

class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs,style=style)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))

folder_path_w = widgets.Text(
    value='20181213',
    placeholder='This Notebook Directory/...',
    description='Folder path',
    disabled=False,
    style=style
)
start_num_w = widgets.IntText(
    value=20,
    description='Starting File',
    disabled=False,
    style=style
)
stop_num_w = widgets.IntText(
    value=63,
    description='Ending File',
    disabled=False,
    style=style
)
start_int_w = widgets.FloatText(
    value=0.00,
    description='Integration Start (ms)',
    disabled=False,
    style=style
)
stop_int_w = widgets.FloatText(
    value=4,
    description='Integration End (ms)',
    disabled=False,
    style=style
)
print_w = widgets.Checkbox(
    value=False,
    description='Print Output?',
    disabled=False
)
indep_var_w = widgets.Text(
    placeholder='Separated by commas',
    description='Independent Variable',
    disabled=False,
    style=style
)


def wrapper_getRawDataset(w):
        print('Generating raw data from ',folder_path_w.value)
        #results = processData(folder_path_w.value,start_num_w.value,stop_num_w.value,start_setpnt_w.value,stop_setpnt_w.value,setpnt_int_w.value,print_w.value)
        results = getRawDataset(folder_path_w.value,start_num_w.value,stop_num_w.value,print_w.value)
        w.value = results
        print('Done')

def wrapper_calcODs(w):
        print('Calculating ODs from ',folder_path_w.value)
        #results = processData(folder_path_w.value,start_num_w.value,stop_num_w.value,start_setpnt_w.value,stop_setpnt_w.value,setpnt_int_w.value,print_w.value)
        results = calculateODs(*rawdata_lb.value, print_bool=print_w.value)
        w.value = results
        print('Done')
        return

def wrapper_intODs(w):
        print('Integrating and processing ODs from ',folder_path_w.value)
        indep_var = [float(num_string.strip()) for num_string in indep_var_w.value.split(',')]
        results = integrateODs(*calcODs_lb.value, indep_var = indep_var_w.value, start_int=start_int_w.value,stop_int=stop_int_w.value, print_bool=print_w.value)
        w.value = results
        print('Done')
        return

def widget_layout():
    rawdata_lb = LoadedButton(description="Get Raw Dataset", value=[])
    rawdata_lb.on_click(wrapper_getRawDataset)
    calcODs_lb = LoadedButton(description="Calculate ODs", value=[])
    calcODs_lb.on_click(wrapper_calcODs)
    intODs_lb = LoadedButton(description="Integrate ODs", value=[])
    intODs_lb.on_click(wrapper_intODs)
    r0=widgets.HBox([folder_path_w])
    r1 = widgets.HBox([start_num_w,stop_num_w])
    r2 = widgets.HBox([rawdata_lb,print_w])
    r3 = widgets.HBox([calcODs_lb,print_w])
    r4a = widgets.HBox([indep_var_w])
    r4b = widgets.HBox([start_int_w,stop_int_w])
    r5 = widgets.HBox([intODs_lb,print_w])
    #display(widgets.VBox([r0,r1,r2,r3,r4a,r4b,r5]))
    raw_settings = widgets.Accordion(children=[r1])
    raw_settings.set_title(0, 'Raw Data Settings')
    display(raw_settings)
    display(widgets.VBox([r2,r3]))
    analysis_settings = widgets.Accordion(children=[widgets.VBox([r4a,r4b])])
    analysis_settings.set_title(0, 'Analysis Settings')
    display(analysis_settings)
    display(r5)

def run():
    widget_layout()
    return
