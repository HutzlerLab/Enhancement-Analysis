import pathlib
import numpy as np
import csv
from scipy.signal import savgol_filter

dt_INDEX = 0
start_INDEX = 1
nsample_INDEX = 2
trigtime_INDEX = 3
blocked_INDEX = 4
param_LENGTH = 5


############################################# Raw Data Functions #############################################

##Main Functions##

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



##Sub-functions##


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
    return [data,parameters]

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
