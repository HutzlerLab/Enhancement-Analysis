import pathlib
import numpy as np
import csv
import matplotlib.pyplot as plt
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
def time_array(parameters):
    dt = parameters[dt_INDEX]
    t0 = parameters[start_INDEX]
    npnts = parameters[nsample_INDEX]
    time_ms = np.round(np.linspace(t0,dt*(npnts-1)+t0,npnts),decimals = 6)
    return time_ms

'''Calculate the optical depth (OD) signal from a single text file.
    Returns a dataset, which consists of the optical dpeth and acquisition parameters'''
def calculateSingleOD(root_folder,num,ch=3):
    raw_traces,params = file2dataset(root_folder,num)
    if ch == 3: #Get both channels
        ch1 = raw2OD(np.array(raw_traces[0]),time_array(params))
        ch2 = raw2OD(np.array(raw_traces[1]),time_array(params))
        ODs = [ch1,ch2]
    else:
        single_ch = raw2OD(np.array(raw_traces[ch-1]),time_array(params))
        ODs = [single_ch]
    return [ODs,params]

'''Calculate the optical depth (OD) signal from a single text file.
    Returns a dataset, which consists of the optical dpeth and acquisition parameters'''
def calculateSingleFluor(root_folder,num,ch=3):
    raw_traces,params = file2dataset(root_folder,num)
    if ch == 3: #Get both channels
        ch1 = raw2fluor(np.array(raw_traces[0]),time_array(params))
        ch2 = raw2fluor(np.array(raw_traces[1]),time_array(params))
        fluor = [ch1,ch2]
    else:
        single_ch = raw2fluor(np.array(raw_traces[ch-1]),time_array(params))
        fluor = [single_ch]
    return [fluor,params]

def calculate_single_OD_fluor(root_folder,num,fluor_ch=1,OD_ch=2):
    raw_traces,params = file2dataset(root_folder,num)
    time_ms = time_array(params)
    abs_OD = raw2OD(raw_traces[OD_ch-1],time_ms)
    fluor = raw2fluor(raw_traces[fluor_ch-1],time_ms)
    return [[fluor,abs_OD],params]


##Sub-functions##


'''Read/write rows from/to CSV'''
def writeCSVrows(array_of_arrays,filepath):
    with open(filepath,'w+',newline='') as f:
        w = csv.writer(f)
        for array in array_of_arrays:
            w.writerow(array)
    print('Done! Array written to', filepath)
    return

def processed_data_filepath(path):
    current = pathlib.Path.cwd()
    data_dir = current.parents[0]
    processed_dir = data_dir / 'Processed_Data' / path
    return processed_dir

def write_labeled_CSV_rows(data_array,file_path,label_array):
    full_path = processed_data_filepath(file_path)
    data_array = np.array(data_array)
    save_all = []
    save_all.append(label_array)
    for _data in data_array.T:
        save_all.append(_data)
    writeCSVrows(save_all,full_path)

def isFloat(value):
    try:
        float(value)
        return True
    except:
        return False

def readCSVrows(file_path,read_header=0,obj=False):
    data_type = 'float'
    if obj:
        data_type='object'
    full_path = processed_data_filepath(file_path)
    header = []
    with open(full_path,'r') as f:
        r = csv.reader(f)
        for i in range(read_header):
            header.append(next(r))
        rows = np.array([np.array([float(value) if isFloat(value) else value for value in row],dtype=data_type) for row in r])
    print('Done! Array read from', file_path)
    return rows,header

def readCSVcolumns(file_path,read_header=0,obj=False):
    data_type = 'float'
    if obj:
        data_type='object'
    full_path = processed_data_filepath(file_path)
    header = []
    with open(full_path,'r') as f:
        r = csv.reader(f)
        for i in range(read_header):
            header.append(next(r))
        rows = np.array([np.array([float(value) if isFloat(value) else value for value in row],dtype=data_type) for row in r])
    columns = rows.T
    print('Done! Array read from', full_path)
    return columns,header

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
    except OSError:
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
def genPath(folder,num,filename='spectra_',data_folder='Raw_Data'):
    num_str = str(num)
    file =  filename + num_str + '.txt'
    #path is given from current working directory
    cwd = pathlib.Path.cwd()
    data_dir = cwd.parents[0] / data_folder
    filepath = data_dir / folder / file
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
    floor = smoothed_data[smoothed_data>0].min()
    smoothed_data[smoothed_data<0] = floor
    #Calculate OD, fix floating point errors
    OD = np.log(offset/smoothed_data)
    if OD[trigger_index]<0:
        offset = smoothed_data[trigger_index]
        OD = np.log(offset/smoothed_data)
    return OD

def raw2fluor(raw_data,time_ms):
    trigger_index = np.searchsorted(time_ms,0)
    beforeYAG_index = np.searchsorted(time_ms,-0.1)
    #Calculate DC offset, convert signal to OD
    offset = raw_data[:beforeYAG_index].mean()
    #Smooth the data
    smoothed_data = smooth(raw_data,window=60)
    smoothed_data-=offset
    #Calculate OD, fix floating point errors
    return smoothed_data


'''Function for smoothing data. Currently uses Savitzky-Golay filter, which fits a window
    onto a polynomial of some order, and then uses the polynomial to estimate the value'''
def smooth(data,window=5,poly_order=3):
    #window value must be odd
    if window % 2 == 0:
        window+=1
    smoothed_data = savgol_filter(data, window, poly_order)
    return smoothed_data

'''Iterate through a series of files and compiles ODs and parameters.
    Labels the whole series as blocked (True) or unblocked (False).
    Returns an array of ODs and parameters'''
def labelBUB(param_array,blocked_bool):
    for single_param in param_array:
        single_param[blocked_INDEX] = blocked_bool
    return [ODs,params]

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


'''Extracts trigger timestamps from 2 (blocked/unblocked) parameter arrays,
    relative to whichever was first'''
def extractTimestampsBUB(blocked_params,unblocked_params):
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
