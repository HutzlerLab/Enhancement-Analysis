import pathlib
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev

dt_INDEX = 0
start_INDEX = 1
nsample_INDEX = 2
trigtime_INDEX = 3
blocked_INDEX = 4
param_LENGTH = 5


############################################# Raw Data Functions #############################################

##Main Functions##

def time_array(parameters):
    '''This function generates a time array using metadata, supplied as the
    function input. Outputs numpy array of time in miliseconds.'''
    dt = parameters[dt_INDEX]
    t0 = parameters[start_INDEX]
    npnts = parameters[nsample_INDEX]
    time_ms = np.round(np.linspace(t0,dt*(npnts-1)+t0,npnts),decimals = 6)
    return time_ms

def process_single_datafile(root_folder, file_num, channel_types):
    '''This function processes a single raw data file. It returns the
    processed data and the parameters associated with that data.

    Inputs:
    root_folder: path for data file from current working directory.
    file_num: number labeling data file
    channel_types: a list containing the data types that label each channel.
    For example, if channel 1 is absorption and channel 2 fluorescence, then
    the channel_type is ['abs','fluor']

    Outputs: [processed_traces, params]
    processed_traces: an array of the processed data read from the file.
    Same order as channel_types.
    params: metadata associated with the data file.
    '''
    raw_traces, params = file2dataset(root_folder,file_num)
    time_ms = time_array(params)
    processed_traces = []
    for ch,type in enumerate(channel_types):
        if type == 'abs':
            abs_trace = raw2OD_wLine(raw_traces[ch], time_ms)
            processed_traces.append(abs_trace)
        elif type == 'fluor':
            fluor_trace = raw2fluor_wLine(raw_traces[ch],time_ms)
            processed_traces.append(fluor_trace)
    return [processed_traces, params]

def write_labeled_CSV_rows(data_array,file_path,labels):
    '''Wrapper for write_CSV_rows. Writes data arrays to location in file path.
    Also labels each row.

    Inputs:
    data_array: containing all arrays to be written to
    file_path: location of file to be written, referenced to parent directory
    of current working directory
    labels: list of labels for each data array to be written
    '''
    full_path = gen_filepath_from_parent(file_path)
    data_array = np.array(data_array)
    save_all = []
    save_all.append(labels)
    for _data in data_array.T:
        save_all.append(_data)
    writeCSVrows(save_all,full_path)
    return

def read_CSV_rows(file_path,read_header=0,obj=False):
    '''Reads rows of data from CSV file. Returns [data,header]

    Inputs:
    file_path: location of file, referenced from parent directory
    read_header: number of lines to read as header instead of as data.
    Default is 0.
    obj: to typecast the read data as an object, set to True. Default is False.
    '''
    data_type = 'float'
    if obj:
        data_type='object'
    full_path = gen_filepath_from_parent(file_path)
    header = []
    with open(full_path,'r') as f:
        r = csv.reader(f)
        for i in range(read_header):
            header.append(next(r))
        rows = np.array([np.array([float(value) if isFloat(value) else value for value in row],dtype=data_type) for row in r])
    print('Done! Array read from', file_path)
    return rows,header

def read_CSV_columns(file_path,read_header=0,obj=False):
    '''Reads columns of data from CSV file. Returns [data,header]

    Inputs:
    file_path: location of file, referenced from parent directory
    read_header: number of lines to read as header instead of as data.
    Default is 0.
    obj: to typecast the read data as an object, set to True. Default is False.
    '''
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


##Sub-functions##


def write_CSV_rows(data_arrays,filepath):
    '''This function writes arrays in rows to a CSV file.

    Inputs:
    data_arrays: contains all arrays to be written
    filepath: path for CSV file
    '''
    with open(filepath,'w+',newline='') as f:
        w = csv.writer(f)
        for array in data_arrays:
            w.writerow(array)
    print('Done! Array written to', filepath)
    return

def gen_filepath_from_parent(path):
    '''Generates path to a file located in a parellel directory
    from the current working directory (cwd). Specifically, if the cwd is in
    parent_dir/cwd, then it generates a path to parent_dir/path
    '''
    current = pathlib.Path.cwd()
    parrent_dir = current.parents[0]
    new_path = parent_dir / path
    return new_path

def isFloat(value):
    '''Checks if input is float. Probably easier to use isinstance.'''
    try:
        float(value)
        return True
    except:
        return False

def import_raw_data_Cleverscope(filepath,usecols=(1,2)):
    '''Imports raw traces from data columns in single cleverscope text file.
    Input is filepath.'''
    try:
        arr = np.genfromtxt(filepath,delimiter='', skip_header = 15, usecols=usecols)
        ch1_raw = arr.T[0] #Channel A
        ch2_raw = arr.T[1] #Channel B
        raw_data = [ch1_raw,ch2_raw]
    except ValueError:
        print('Issues with file located at: ',filepath)
        raw_data = [np.zeros(10000),np.zeros(10000)]
    except OSError:
        print('Issues with file located at: ',filepath)
        raw_data = [np.zeros(10000),np.zeros(10000)]
    return raw_data

def import_params_Cleverscope(filepath):
    '''Import metadata parameters from Cleverscope file.

    Parameters returned as list:
    dt: time step of data traces (ms)
    start: start time of data traces, relative to trigger (ms)
    nsample: number of elements in time series
    trigger time: UNIX time of trigger
    '''
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
            # The time is given as the number of days since 12/29/1899 11:00:00 UTC
            # We can convert this to UNIX by subtracting 2209208400 seconds
            # Ask Arian if you are curious
            time_sec = float(text.split('\t')[1].strip())*24*3600 #convert days to seconds
            time_unix = time_sec - 2209208400
            params[trigtime_INDEX]=time_unix
    return params

def gen_data_path(folder,num,filename='spectra_',data_folder='Raw_Data'):
    '''Generate filepath for single text file named "spectra_num.txt", located
    in a different folder than the current working directorry.

    Inputs:
    folder:
     '''
    num_str = str(num)
    file =  filename + num_str + '.txt'
    #path is given from current working directory
    path = gen_filepath_from_parent(data_folder)
    filepath = path / file
    return filepath

'''From a single file location, obtain dataset with raw data and associated parameters'''
def file2dataset(root_folder,num,print_bool=False):
    filepath = gen_data_path(root_folder,num)
    if print_bool:
        print(filepath)
    data = import_raw_data_cleverscope(filepath)
    parameters = import_params_Cleverscope(filepath)
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
    offset = (raw_data[:beforeYAG_index].mean()+raw_data[-1000].mean())/2
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
def extractTimestamps(params,subtract_offset=True):
    timestamps = []
    if subtract_offset:
        for single_params in params:
            if len(timestamps)==0:
                initial = single_params[trigtime_INDEX]
                timestamps.append(0)
            else:
                elapsed = single_params[trigtime_INDEX] - initial
                timestamps.append(elapsed)
    else:
            for single_params in params:
                unix_time = single_params[trigtime_INDEX]
                timestamps.append(unix_time)
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

def auto_integrate(OD,time_ms,start=0,fig_num=1):
    start_i = np.searchsorted(time_ms,start)
    smoothed_deriv = savgol_filter(OD,3001,5,deriv=1)
    smoothed_deriv/=smoothed_deriv.max()
    norm_OD = OD/OD.max()
    cut_i = np.argmin(smoothed_deriv[start_i:-500])+start_i #ignore before trigger and last few points
    OD_cut = norm_OD[cut_i:]
    smoothed_deriv_cut = smoothed_deriv[cut_i:]
    time_cut = time_ms[cut_i:]
    if OD.max() < 0.005:
        cut = 0.5
    else:
        cut = 0.03
    OD_threshold = cut
    deriv_threshold = abs(smoothed_deriv[cut_i:]).max()*0.03
    still_looking=True
    deriv_check = False
    OD_check = False
    for i in range(len(time_cut)):
        OD_val = abs(OD_cut[i])
        deriv_val = abs(smoothed_deriv_cut[i])
        if deriv_val < deriv_threshold:
            deriv_check = True
        if OD_val < OD_threshold:
            OD_check = True
        if deriv_check and OD_check:
            stop = time_cut[i]
            still_looking=False
            break
    if still_looking:
        print('Cannot find limit')
        plt.figure(fig_num+15)
        plt.title('Problem Traces')
        plt.plot(time_ms,OD)
        plt.figure(fig_num+16)
        plt.title('Problem Smoothed Traces')
        #plt.plot(time_cut, smoothed_OD_cut)
        plt.plot(time_cut,smoothed_deriv_cut)
        stop = 0.5
    stop_i = np.searchsorted(time_ms,stop)
    t = slice(start_i,stop_i)
    dt = np.round(time_ms[1]-time_ms[0],decimals=6)
    integrated = np.round(OD[t].sum()*dt,decimals=6)
    plt.figure(fig_num)
    plt.plot(time_ms[t],OD[t])
    return integrated

# Old Functions #

'''Calculate the optical depth (OD) signal from a single text file.
    Returns a dataset, which consists of the optical dpeth and acquisition parameters'''
def calculateSingleOD(root_folder,num,ch=3):
    raw_traces,params = file2dataset(root_folder,num)
    if ch == 3: #Get both channels
        ch1 = raw2OD_wLine(np.array(raw_traces[0]),time_array(params))
        ch2 = raw2OD_wLine(np.array(raw_traces[1]),time_array(params))
        ODs = [ch1,ch2]
    else:
        single_ch = raw2OD_wLine(np.array(raw_traces[ch-1]),time_array(params))
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
    abs_OD = raw2OD_wLine(raw_traces[OD_ch-1],time_ms)
    fluor = raw2fluor_wLine(raw_traces[fluor_ch-1],time_ms)
    return [[fluor,abs_OD],params]


# def auto_integrate(OD,time_ms,start=0,fig_num=1):
#     start_i = np.searchsorted(time_ms,start)
#     f = splrep(time_ms,OD,k=4,s=0.1)
#     f_d1 = splev(time_ms,f,der=1)
#     f_d2 = splev(time_ms,f,der=2)
#     after_max_i = np.argmin(f_d1[start_i+10:-100])+start_i+10 #ignore before trigger and last few points
#     f_d1_cut = f_d1[after_max_i:]
#     f_d2_cut = f_d2[after_max_i:]
#     time_cut = time_ms[after_max_i:]
#     still_looking = True
#     d1_thresh = abs(OD).max()*0.01
#     d2_thresh = abs(f_d1_cut).max()*0.01
#     for i in range(len(time_cut)):
#         d1 = abs(f_d1_cut[i])
#         d2 = abs(f_d2_cut[i])
#         if d1 < d1_thresh and d2 < d2_thresh:
#             stop = time_cut[i]
#             still_looking=False
#             break
#     if still_looking:
#         print('Issue integrating the following trace')
#         plt.figure(fig_num+3)
#         plt.plot(time_ms,OD)
#         plt.plot(time_ms,splev(time_ms,f))
#         plt.plot(time_ms,f_d1)
#         stop = 0.1
#     stop_i = np.searchsorted(time_ms,stop)
#     t = slice(start_i,stop_i)
#     dt = np.round(time_ms[1]-time_ms[0],decimals=6)
#     integrated = np.round(OD[t].sum()*dt,decimals=6)
#     if still_looking:
#         integrated = np.nan
#     plt.figure(fig_num)
#     plt.plot(time_ms[t],OD[t])
#     return integrated


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


def line_func(x, A, B):
    return A*x + B

def raw2OD_wLine(raw_data,time_ms):
    trigger_index = np.searchsorted(time_ms,0)
    beforeYAG_index = np.searchsorted(time_ms,-0.1)
    after_abs_index = np.searchsorted(time_ms,time_ms[-1000])
    #Calculate linear and DC offset, convert signal to OD
    fit_time = np.concatenate((time_ms[:beforeYAG_index],time_ms[after_abs_index:]))
    fit_data = np.concatenate((raw_data[:beforeYAG_index],raw_data[after_abs_index:]))
    # time_list=np.ndarray.tolist(time_ms)
    # data_list=np.ndarray.tolist(raw_data)
    # fit_time=time_list[:beforeYAG_index]
    # fit_time.extend(time_list[after_abs_index:])
    # fit_time=np.array(fit_time)
    # fit_data=data_list[:beforeYAG_index]
    # fit_data.extend(data_list[after_abs_index:])
    # fit_data=np.array(fit_data)
    #popt, pcov = curve_fit(line_func, time_ms[:beforeYAG_index], raw_data[:beforeYAG_index])
    #popt, pcov = curve_fit(line_func, time_ms[after_abs_index:], raw_data[after_abs_index:])
    popt, pcov = curve_fit(line_func, fit_time , fit_data,p0=[(fit_data[-1]-fit_data[0])/(time_ms[-1]-time_ms[0]),fit_data.mean()])
    A=popt[0]
    offset=popt[1]
    flat_data=np.zeros(len(raw_data))
    for i in range(len(raw_data)):
        flat_data[i]=raw_data[i]-(line_func(time_ms[i],A,offset)-offset)
    #offset = raw_data[:beforeYAG_index].mean()
    #Smooth the data
    smoothed_data = smooth(flat_data,window=60)
    floor = smoothed_data[smoothed_data>0].min()
    smoothed_data[smoothed_data<0] = floor

    smoothed_plot = smooth(raw_data,window=60)
    floor = smoothed_plot[smoothed_plot>0].min()
    smoothed_plot[smoothed_plot<0] = floor
    # plt.plot(time_ms,smoothed_plot)
    # plt.plot(time_ms,line_func(time_ms,A,offset))
    #Calculate OD, fix floating point errors
    OD = np.log(offset/smoothed_data)
    if OD[trigger_index]<0:
        offset = smoothed_data[trigger_index]
        OD = np.log(offset/smoothed_data)
    return OD

def raw2fluor_wLine(raw_data,time_ms):
    trigger_index = np.searchsorted(time_ms,0)
    beforeYAG_index = np.searchsorted(time_ms,-0.1)
    after_abs_index = np.searchsorted(time_ms,time_ms[-1000])
    #Calculate linear and DC offset, convert signal to OD
    fit_time = np.concatenate((time_ms[:beforeYAG_index],time_ms[after_abs_index:]))
    fit_data = np.concatenate((raw_data[:beforeYAG_index],raw_data[after_abs_index:]))
    # time_list=np.ndarray.tolist(time_ms)
    # data_list=np.ndarray.tolist(raw_data)
    # fit_time=time_list[:beforeYAG_index]
    # fit_time.extend(time_list[after_abs_index:])
    # fit_time=np.array(fit_time)
    # fit_data=data_list[:beforeYAG_index]
    # fit_data.extend(data_list[after_abs_index:])
    # fit_data=np.array(fit_data)
    #popt, pcov = curve_fit(line_func, time_ms[:beforeYAG_index], raw_data[:beforeYAG_index])
    #popt, pcov = curve_fit(line_func, time_ms[after_abs_index:], raw_data[after_abs_index:])
    popt, pcov = curve_fit(line_func, fit_time , fit_data,p0=[(fit_data[-1]-fit_data[0])/(time_ms[-1]-time_ms[0]),fit_data.mean()])
    A=popt[0]
    offset=popt[1]
    flat_data=np.zeros(len(raw_data))
    for i in range(len(raw_data)):
        flat_data[i]=raw_data[i]-(line_func(time_ms[i],A,offset)-offset)
    #offset = raw_data[:beforeYAG_index].mean()
    #Smooth the data
    smoothed_data = smooth(flat_data,window=60)
    floor = smoothed_data[smoothed_data>0].min()
    smoothed_data[smoothed_data<0] = floor

    offset = (raw_data[:beforeYAG_index].mean()+raw_data[-1000].mean())/2
    #Smooth the data
    smoothed_data-=offset

    return smoothed_data


def calculateSingleOD_wline(root_folder,num,after_abs_time=5,ch=3):
    raw_traces,params = file2dataset(root_folder,num)
    if ch == 3: #Get both channels
        ch1 = raw2OD_wLine(np.array(raw_traces[0]),time_array(params))
        ch2 = raw2OD_wLine(np.array(raw_traces[1]),time_array(params))
        ODs = [ch1,ch2]
    else:
        single_ch = raw2OD_wLine(np.array(raw_traces[ch-1]),time_array(params))
        ODs = [single_ch]
    return [ODs,params]





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
