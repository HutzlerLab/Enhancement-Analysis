'''
Code written for data analysis of signals from a CBGB source, in particular
to deal with chemically enhanced signals.

This code was originally written to analyze data from a digital oscilloscope
("CleverScope"). This code was designed to be run in
a Jupyter Notebook. For details on implementation, open the file
"Sample_Notebook.ipynb" in Jupyter.

Eventually, the goal is to make this code object oriented.

Written by Arian Jadbabaie
Email: arianjad@gmail.com
'''

import pathlib
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev
from scipy.fftpack import rfftfreq, rfft,irfft
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.optimize import curve_fit
import multiprocessing as mp
from functools import partial
import sys
from copy import deepcopy

'''Each data trace has associated with it some metadata, referred to as
"parameters". These parameters are stored in a dict. The elements are:

    dt: time step of data traces (ms)
    start: start time of data traces, relative to trigger (ms)
    nsample: number of elements in time series
    trigtime: UNIX time of trigger
'''

##Main Functions##

def integrate_data_array(data_array,metadata_array,bounds_array,plot=False):
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(mp_integrate, [(data,meta,bounds_array) for data,meta in zip(data_array,metadata_array)])
    pool.close()
    pool.join()
    integrated_data_array = np.array(results)
    if plot:
        plot_slices(data_array,metadata_array,bounds_array)
    return integrated_data_array.T #sort by channel instead of by run

def mp_integrate(data,meta,bounds_array):
    time_ms = time_array(meta)
    dt = meta['dt']
    integrated_data = np.zeros(data.shape[0])
    for i,(channel, bounds) in enumerate(zip(data,bounds_array)):
        start_i = np.searchsorted(time_ms,bounds[0])
        stop_i = np.searchsorted(time_ms,bounds[1])
        integrated_data[i] = np.round(channel[start_i:stop_i].sum()*dt,decimals=6)
    return integrated_data


def extract_frequency(metadata):
    return np.array([meta['frequency'] for meta in metadata])

def slice_integrate(data,metadata,start_stop,plot,fig_num):
    time_ms = time_array(metadata)
    start_i = np.searchsorted(time_ms,start_stop[0])
    stop_i = np.searchsorted(time_ms,start_stop[1])
    t = slice(start_i,stop_i)
    dt = np.round(time_ms[1]-time_ms[0],decimals=6)
    integrated = np.round(data[t].sum()*dt,decimals=6)
    if plot:
            plt.figure(fig_num)
            plt.plot(time_ms[t],data[t])
    return integrated

def plot_slices(data_array,meta_data_array,bounds_array):
    time_ms = time_array(meta_data_array[0])
    for i in range(data_array.shape[1]):
        start_i = np.searchsorted(time_ms,bounds_array[i][0])
        stop_i = np.searchsorted(time_ms,bounds_array[i][1])
        t = slice(start_i,stop_i)
        plt.figure(i)
        plt.title('Channel {}'.format(i+1))
        plt.xlabel('Time After Ablation (ms)')
        plt.ylabel('Signal')
        for j in range(data_array.shape[0]):
            plt.plot(time_ms[t],data_array[j,i,t])
    return




def read_raw_data_array(file_numbers,file_name,folder_path,DAQ='PXI',version=4,map_version='starmap',chunksize=1):#channel_types,smooth=False,abs_filter=1500,fluor_filter=2000,avg=0,version=1,avg_cycle=False):
    '''This function is used to convert a series of raw data files to
    processed data traces. The raw data can be in the form of absorption
    data, or fluorescence data.

    Inputs:
    folder: local path of folder containing raw data
    file_numbers: array containing the file numbers to process
    channel_types: a list containing the data types that label each channel.
    For example, if channel 1 is absorption and channel 2 fluorescence, then
    the channel_type is ['abs','fluor']
    root_folder: parent folder containing data folder
    file_name: name of data file. The format should be "nameXXX.txt", where
    XXX is a number labeling the file
    version: used to implement backwards compatability.
    -1 = before COVID shutdown, 0 = first runs during shutdown,

    Outputs:
    A list of the form [processed_data,param_array]
    Each element in data_array and parameter_array correspond to the traces and
    metadata obtained from a single file.
    '''
    pool = mp.Pool(mp.cpu_count())
    worker = partial(read_raw_file,file_name=file_name,folder_path = folder_path,DAQ=DAQ,version=version)
    if map_version=='starmap':
    	results = pool.starmap(worker,[(num,) for num in file_numbers],chunksize=chunksize)
    elif map_version=='imap':
    	results = pool.imap(worker, file_numbers,chunksize=chunksize)
    elif map_version == 'map':
    	results = pool.map(worker, file_numbers,chunksize=chunksize)
    pool.close()
    pool.join()
    raw_data_array,param_array = list(zip(*results))
    # for i,num in enumerate(file_numbers):
    #     if avg == 0:
    #         avg_files = 1
    #     elif i > len(file_numbers) - avg:
    #         avg_files = len(file_numbers)-i
    #     else:
    #         avg_files=avg
    #     processed_data, params = process_single_data_file(folder_path, num, file_name, channel_types,smooth=smooth,abs_filter=abs_filter,fluor_filter=fluor_filter,avg_files=avg_files,last = last,version=version)
    #     param_array[i] = params
    #     data_array[i] = processed_data
    # data_array = np.array(data_array) # decided not to turn these into numpy arrays
    # param_array = np.array(param_array)
    return [np.array(raw_data_array),param_array]

def correct_freq(param_array,laser=None):
    all_freq = np.array([np.array(meta['wavemeter readings']) for meta in param_array]).T
    for laser_index in range(len(all_freq)):
        if laser==None:
            pass
        elif laser != laser_index+1:
            continue
        frequency = all_freq[laser_index]
        exposure = False
        for i in range(len(frequency)):
            if frequency[i]<0:
                if not exposure:
                    print('Wavemeter Exposure Issue')
                    exposure=True
                else:
                    pass
                j = i
                k = i
                while (j>0 and k<(len(frequency)-1)) and (frequency[j-1]<0 or frequency[k+1]<0):
                    if frequency[j-1]<0:
                        j-=1
                    if frequency[k+1]<0:
                       k+=1
                if j<= 0 or k>=(len(frequency)-1):
                    print('Wavemeter issue at endpoints')
                else:
                    frequency[i]=(frequency[k+1]-frequency[j-1])/((k+1)-(j-1))*(i-(j-1))+frequency[j-1]
        # freq = frequency[frequency>0]
        # slope = (freq[-1] - freq[0])/len(freq)
        # for i in range(len(frequency)):
        #     if i==0 and frequency[i]<0:
        #         j = 1
        #         while frequency[j]<0:
        #             j+=1
        #         for k in range(j):
        #             frequency[j-k-1] = frequency[j-k]-slope
        #     elif frequency[i]<0:
        #         frequency[i] = frequency[i-1]+slope
        all_freq[laser_index] = frequency
    for i in range(len(param_array)):
        param_array[i]['wavemeter readings'] = all_freq.T[i]
    return param_array

# def handle_averaging(file_numbers,avg,folder_path,num,file_name,avg_cycle):
#     if (avg == 0 or avg == 1) and avg_cycle == False:
#         return file_numbers
#     elif avg>1 and avg_cycle == False:
#         cmd_files = []
#         i0 = file_numbers[0]
#         params_0 = read_raw_file(folder_path,i0,file_name,DAQ=DAQ,version=version,meta_only=True)
#         #Need to find out how many command files there are. If no averaging we can just step through. If there is averaging we have to be careful with stepping
#         #total reps tells us how many averages the DAQ took. So if no averaging, total reps = 1, if averaging, total reps > 1
#         total_reps = params_0['total repetition']
#         cmd_files.append(params_0['command file'])
#         repeat_cmd = False
#         i=total_reps
#         while repeat_cmd==False:
#             _params = read_raw_file(folder_path,i0+i,file_name,DAQ=DAQ,version=version,meta_only=True)
#             if _params['command file'] in cmd_files:
#                 repeat_cmd = True
#             else:
#                 cmd_files.append(_params['command file'])
#                 i+=total_reps
#         #Now we find all the data files associated with each cmd file
#         pool = mp.Pool(mp.cpu_count())
#         worker = mp_sort_cmd_files(folder_path,num,file_name,cmd_files,DAQ=DAQ, verison=version)


def mp_process_single_data_file(file_num,file_name,folder_path, DAQ='PXI',version=2):
    # if avg == 0 and avg_cycle == False:
    #     avg_files = 1
    # elif i > len(file_numbers) - avg:
    #     avg_files = len(file_numbers)-i
    # elif avg_cycle==False:
    #     avg_files=avg

    raw_data, params = read_raw_file(file_num,file_name,folder_path,DAQ=DAQ,version=version)
    return np.array([raw_data,params])


def process_single_data_file(file_num, file_name, folder_path, DAQ='PXI',version=2):
    '''This function processes a single raw data file. It returns the
    processed data and the parameters associated with that data.

    Inputs:
    data_folder: name of folder containing data file, string.
    file_num: number labeling data file
    channel_types: a list containing the data types that label each channel.
    For example, if channel 1 is absorption and channel 2 fluorescence, then
    the channel_type is ['abs','fluor']
    root_folder: parent folder containing data folder
    file_name: name of data file. The format should be "nameXXX.txt", where
    XXX is a number labeling the file

    Outputs: [processed_traces, params]
    processed_traces: an array of the processed data read from the file.
    Same order as channel_types.
    params: metadata associated with the data file.
    '''
    raw_traces, params = read_raw_file(file_num,file_name,folder_path,DAQ=DAQ,version=version)

    # time_ms = time_array(params)
    # processed_traces = [[] for _ in range(len(channel_types))]
    # for ch,type in enumerate(channel_types):
    #     if type == 'abs':
    #         abs_trace = process_raw_abs(raw_traces[ch], time_ms,plot=plot,smooth=smooth,filter=abs_filter)
    #         processed_traces[ch] =abs_trace
    #     elif type == 'fluor':
    #         fluor_trace = process_raw_fluor(raw_traces[ch],time_ms,plot=plot,smooth=smooth,filter=fluor_filter)
    #         processed_traces[ch] = fluor_trace
    return [raw_traces, params]

def sort_by_command(data_array,meta_dicts):
    waveform_dict = {}
    for i,meta in enumerate(meta_dicts):
        try:
            waveform_dict[meta['command file']][1].append(meta_dicts[i])
            waveform_dict[meta['command file']][0].append(data_array[i])
        except KeyError:
            waveform_dict[meta['command file']] = [[data_array[i]],[meta_dicts[i]]]
    for key in waveform_dict:
        waveform_dict[key][0] = np.array(waveform_dict[key][0])
        print('Found {} shots with command file {}'.format(len(waveform_dict[key][1]),key))
    return waveform_dict


def average_raw_data_array(raw_data,params,num_avg=1,cycle_avg=False):
    pool = mp.Pool(mp.cpu_count())
    #Need to handle end points. Easiest thing to do is to chop them off.
    if cycle_avg:
        num_avg = params[0]['total cycle']
        cycle_start = params[0]['cycle number']
        cycle_end = params[-1]['cycle number']
        i=0
        j = len(params)
        while cycle_start % num_avg != 1:
            i+=1
            cycle_start = params[i]['cycle number']
        while cycle_end % num_avg != 0:
            j-=1
            cycle_end = params[j]['cycle number']
    else:
        i=0
        j=len(params)
        while j+1 % num_avg != 0:
            j-=1
    raw_data=raw_data[i:j]
    params = params[i:j]
    #
    #     if params[0]['cycle number'] % total
    #     while()
    # for i, meta in enumerate(params):
    #     if cycle_avg:
    #         cycle_num = meta['cycle number']
    #     else:
    #         cycle_num = i+1
    #     if cycle_num % num_avg == 0 and len(cycle_indices)>0:
    #         average_indices.append(cycle_indices)
    #         cycle_indices = []
    #     else:
    #         cycle_indices.append(i)
    #     if len(cycle_indices)>0:
    #         avg_indices.append(cycle_indices)
    #partial(mp_process_single_data_file,DAQ=DAQ,version=version)
    results = pool.starmap(mp_avg, [(raw_data[run_index:run_index+num_avg], params[run_index:run_index+num_avg]) for run_index in range(0,len(params),num_avg)])
    pool.close()
    pool.join()
    avg_data_array,avg_param_array = list(zip(*results))
    return [np.array(avg_data_array), avg_param_array]

def process_raw_data_array(data_array,param_array,ch_types=None):
    pool = mp.Pool(mp.cpu_count())
    mp_process_raw_partial = partial(mp_process_raw,channel_types=ch_types)
    results = pool.starmap(mp_process_raw_partial, [(data,params) for data,params in zip(data_array,param_array)])
    pool.close()
    pool.join()
    proc_data_array,param_array = list(zip(*results))
    return [np.array(proc_data_array),param_array]

def mp_process_raw(data,params,plot=False,smooth=True,channel_types = None,abs_filter = 3000, fluor_filter = 3000):
    time_ms = time_array(params)
    if channel_types == None:
        channel_types = params['channel types']
    processed_traces = [[] for ch in channel_types]
    for ch,type in enumerate(channel_types):
        if type == 'abs':
            abs_trace = process_raw_abs(data[ch], time_ms,plot=plot,smooth=smooth,filter=abs_filter)
            processed_traces[ch] = np.array(abs_trace)
        elif type == 'fluor':
            fluor_trace = process_raw_fluor(data[ch],time_ms,plot=plot,smooth=smooth,filter=fluor_filter)
            processed_traces[ch] = np.array(fluor_trace)
        elif type == 'i2' or type == 'iod':
            processed_traces[ch] = data[ch]
    return [np.array(processed_traces),params]



def mp_avg(data_array, param_array):
    avg_data = np.zeros((data_array.shape[1:]))
    avg_data = np.mean(data_array,axis=0)

    avg_param = deepcopy(param_array[0])
    avg_wavemeter = np.zeros(len(avg_param['wavemeter readings']))
    avg_time = 0
    n = len(param_array)
    for _params in param_array:
        avg_wavemeter+= np.array(_params['wavemeter readings'])
        avg_time+=_params['trigtime']
    avg_param['wavemeter readings'] = avg_wavemeter/n
    avg_param['trigtime'] = avg_time/n
    return [avg_data,avg_param]


def time_array(parameters):
    '''This function generates a time array using metadata, supplied as the
    function input. Outputs a numpy array of time in miliseconds.'''
    dt = parameters['dt']
    t0 = parameters['start']
    npnts = parameters['nsample']
    time_ms = np.round(np.linspace(t0,dt*(npnts-1)+t0,npnts),decimals = 6)
    time_ms-=parameters['YAGtrig']
    return time_ms

def write_CSV_rows(data_array,file_path,labels):
    '''Wrapper for write_CSV_rows. Writes data arrays to location in file path.
    Also labels each row.

    Inputs:
    data_array: containing all arrays to be written to
    file_path: location of file to be written, referenced to parent directory
    of current working directory
    labels: list of labels for each data array to be written
    '''
    full_path = pathlib.Path(file_path)
    full_path = str(full_path)
    data_array = np.array(data_array)
    save_all = []
    save_all.append(labels)
    for _data in data_array.T:
        save_all.append(_data)
    _write_CSV_rows(save_all,full_path)
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
    header = []
    with open(file_path,'r') as f:
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
    header = []
    with open(file_path,'r') as f:
        r = csv.reader(f)
        for i in range(read_header):
            header.append(next(r))
        rows = np.array([np.array([float(value) if isFloat(value) else value for value in row],dtype=data_type) for row in r])
    columns = rows.T
    print('Done! Array read from', file_path)
    return columns,header


##################### Data Processing Subfunctions ############################

def process_raw_fluor(raw_data,time_ms,line_background=True,plot=False,smooth=False,filter=1000):
    if line_background:
        processed_trace = raw2fluor_wLine(raw_data,time_ms,plot,smooth,filter)
    else:
        processed_trace = raw2fluor(raw_data,time_ms)
    return processed_trace

def process_raw_abs(raw_data,time_ms,line_background=True,plot=False,smooth=False,filter=1000):
    if line_background:
        processed_trace = raw2OD_wLine(raw_data,time_ms,plot,smooth,filter)
    else:
        processed_trace = raw2OD(raw_data,time_ms)
    return processed_trace



def raw2fluor_wLine(raw_data,time_ms,plot=False,smooth_bool=False,filter=1000):
    #Collect useful indices
    trigger_index = np.searchsorted(time_ms,0)
    beforeYAG_index = np.searchsorted(time_ms,-0.1)
    after_abs_index = np.searchsorted(time_ms,(time_ms[-1]-1))

    #Filter data. Currently just a lowpass butterworth filter
    if smooth_bool:
        N = len(time_ms)
        delta_s = (time_ms[1]-time_ms[0])/1000
        fs = np.round(1/delta_s,4)
        lowcut = filter
        order=2
        filtered_data = butter_lowpass_filter(raw_data,lowcut,fs,order=order)
    else:
        filtered_data = raw_data

    #Remove linear drift and find offset
    fit_time = np.concatenate((time_ms[:beforeYAG_index],time_ms[after_abs_index:]))
    fit_data = np.concatenate((filtered_data[:beforeYAG_index],filtered_data[after_abs_index:]))
    popt, pcov = curve_fit(line_func, fit_time , fit_data,p0=[(fit_data[-1]-fit_data[0])/(time_ms[-1]-time_ms[0]),fit_data.mean()])
    slope=popt[0]
    intercept=popt[1]
    flat_data = filtered_data-line_func(time_ms,slope,intercept)
    #offset = line_func(time_ms[trigger_index],slope,intercept)
    offset = flat_data[:trigger_index].mean()

    final_data = flat_data - offset
    if plot:
        plt.figure(1)
        plt.plot(time_ms,raw_data)
        plt.plot(time_ms,filtered_data)
        plt.figure(2)
        plt.plot(time_ms,flat_data)
        plt.figure(3)
        plt.plot(time_ms,final_data)
    return final_data


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

def raw2OD_wLine(raw_data,time_ms,plot=False,smooth_bool=False,filter=900):
    #Collect useful indices
    trigger_index = np.searchsorted(time_ms,0)
    beforeYAG_index = np.searchsorted(time_ms,-0.1)
    after_abs_index = np.searchsorted(time_ms,8)

    #Filter data. Currently just a lowpass butterworth filter
    if smooth_bool:
        N = len(time_ms)
        delta_s = (time_ms[1]-time_ms[0])/1000
        fs = np.round(1/delta_s,3)
        lowcut = filter
        order=2
        filtered_data = butter_lowpass_filter(raw_data,lowcut,fs,order=order)
    else:
        filtered_data = raw_data

    #Remove linear drift and find offset
    fit_time = np.concatenate((time_ms[:beforeYAG_index],time_ms[after_abs_index:]))
    fit_data = np.concatenate((filtered_data[:beforeYAG_index],filtered_data[after_abs_index:]))
    popt, pcov = curve_fit(line_func, fit_time , fit_data,p0=[(fit_data[-1]-fit_data[0])/(time_ms[-1]-time_ms[0]),fit_data.mean()])
    slope=popt[0]
    intercept=popt[1]
    flat_data = filtered_data-line_func(time_ms,slope,intercept)+filtered_data[beforeYAG_index]
    #offset = line_func(time_ms[trigger_index],slope,intercept)
    offset = flat_data[:trigger_index].mean()

    OD = np.log(offset/flat_data)
    if plot:
        plt.figure(1)
        plt.plot(time_ms,raw_data)
        plt.plot(time_ms,filtered_data)
        plt.figure(2)
        plt.plot(time_ms,flat_data)
        plt.figure(3)
        plt.plot(time_ms,OD)
    return OD

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='lowpass', output='sos')
    return sos

def butter_lowpass_filter(data, lowcut, fs, order=5):
    sos = butter_lowpass(lowcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

def butter_bandstop(lowstop,highstop,fs,order=5):
    nyq = 0.5 * fs
    low = lowstop / nyq
    high = highstop / nyq
    sos = butter(order, [low,high], analog=False, btype='bandstop', output='sos')
    return sos

def butter_bandstop_filter(data, lowstop, highstop, fs, order=5):
    sos = butter_bandstop(lowstop,highstop,fs,order=order)
    y = sosfiltfilt(sos,data)
    return y

def filt(x,t,plot=False):
    fy,y= spectrum(x, t)
    N = t.size
    if plot:
        plt.subplot(211)
        plt.plot(t,x)
        plt.grid(True)
        plt.subplot(212)
        y_a = np.abs(y)/N*2 # Scale accordingly
        plt.plot(fy, y_a)
        plt.grid(True)
        plt.show()
    y_filt = notch_filter(fy,y)
    x_filt = irfft(y_filt)
    return x_filt

def notch_filter(x,y,range_tuples = [(0.49,0.51),(1.9,2.4),(2.4,3),(3,10),(10,50)],atten = [0,0,0.1,0.01,0.01]):
    trunc = y
    for tup,att in zip(range_tuples,atten):
        (low,high) = tup
        [li, hi] = [np.searchsorted(x,low),np.searchsorted(x,high)]
        trunc[li:hi] = y[li:hi]*att
    return trunc

def spectrum(sig, t):
    fy = rfftfreq(sig.size, d=t[1]-t[0])
    y = rfft(sig)
    return fy,y

def inverse_spectrum(fourier,t):
    iy = irfft(fourier)
    return iy,t

def raw2OD(raw_data,time_ms):
    '''Convert raw absorption data into optical depth'''
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

def read_raw_file(num,file_name,folder_path,DAQ='PXI',version = 1,print_bool=False,meta_only=False):
    '''From a single folder location, obtain dataset with raw data and metadata.
    For full info, see documentation of process_single_data_file().
    '''
    file_path = gen_data_filepath(folder_path,num,file_name,version=version)
    file_path=str(file_path)
    if print_bool:
        print(file_path)
    if DAQ == 'PXI':
        header_lines = how_big_header(file_path,version=version)
        meta = import_metadata_PXI(file_path,header_lines=header_lines,version=version)
        if meta_only:
            return meta
        else:
            raw = import_raw_data_PXI(file_path,header_lines=header_lines,version=version)
            meta['channels'],meta['nsample'] = raw.shape
            return [raw,meta]
    elif DAQ == 'Cleverscope':
        meta = import_metadata_Cleverscope(file_path)
        raw = import_raw_data_Cleverscope(filepath)
        return [raw,meta]

def how_big_header(file_path,version=1,lim=1000):
    if version==3:
        version=2
    with open(file_path, 'r') as f:
        lines=[]
        i=0
        found_the_end = False
        while found_the_end == False:
            lines.append(f.readline().strip('\n'))
            i+=1
            line=lines[-1]
            marker = {-1:'X_Value',0:'time\tPXI1Slot' ,1: '***end of header***',2:'***end of header***',4:'***End Code Header***'}[version]
            if marker in line:
                found_the_end = True
                if (marker=='***end of header***') or (marker=='***End Code Header***'):
                    i+=6
            elif i>lim:
                print('Did not find end of header for',file_path)
                break
    return i



def import_raw_data_PXI(filepath,header_lines = 13,version=1):
    '''Imports raw traces from data columns in single PXI text file.
    Input is file path.'''
    if version==3 or version==4:
        version=2
    idx = {-1:0,0:2,1:2,2:1}[version]
    try:
        arr = np.genfromtxt(filepath,delimiter='', skip_header = header_lines)
        raw_data = np.array([ch for ch in arr.T[idx:]])
    except ValueError:
        print('Issues with file located at: ',filepath)
        raw_data = np.array([np.zeros(10000),np.zeros(10000)])
    except OSError:
        print('Issues with file located at: ',filepath)
        raw_data = np.array([np.zeros(10000),np.zeros(10000)])
    return raw_data

def split_strip(string):
    return string.split(':')[-1].strip(';')

def import_metadata_PXI(filepath,header_lines=13, version=1):
    '''Import metadata parameters from Cleverscope file.

    Parameters returned as dict:
    dt: time step of data traces (ms)
    start: start time of data traces, relative to trigger (ms)
    nsample: number of elements in time series
    trigtime: UNIX time of trigger
    '''
    if version==-1:
        Time_Read = 0
        with open(filepath, 'r') as f:
            lines=[]
            for i in range(header_lines):
                lines.append(f.readline().strip('\n'))
        meta={}
        for text in lines:
            if 'cycle number' in text:
                meta['cycle'] = int(split_strip(text))
            if 'iteration' in text:
                meta['iteration'] = int(split_strip(text))
            if 'command file' in text:
                meta['command file'] = split_strip(text).split('\\')[-1].strip('.txt')
            if 'comment' in text:
                meta['comment'] = text.split(':')[1]
            if 'wavenumber' in text:
                if '(cm-1)(cavity):' in text:
                    meta['frequency_cavity'] = float(text.split('(cm-1)(cavity):')[-1].strip(';'))
                elif '(cm-1)(meter):' in text:
                    meta['frequency_meter'] = float(text.split('(cm-1)(meter):')[-1].strip(';'))
                elif '(cm-1):' in text:
                    meta['frequency'] = float(text.split('(cm-1)')[-1].strip(';'))
            if 'Cavity offset(MHz)' in text:
                meta['cavity_offset'] = float(split_strip(text))
            if 'Channels' in text:
                meta['channels'] = int(text.strip('\t').split('\t')[-1])
            if 'Samples' in text:
                meta['nsample'] = [int(x) for x in text.strip('Samples').strip('\t').split('\t')][0]
            if 'Time\t' in text:
                if Time_Read == 1:
                    H_M_S = text.strip('Time\t').split('\t')[0].split(':')
                    hours_in_sec = float(H_M_S[0])*60*60
                    min_in_sec = float(H_M_S[1])*60
                    sec = float(H_M_S[2])
                    meta['trigtime'] = hours_in_sec+min_in_sec+sec
                    Time_Read+=1
                else:
                    Time_Read+=1
            if 'Delta_X' in text:
                meta['dt'] = [float(x) for x in text.strip('Delta_X').strip('\t').split('\t')][0]*1000
            if 'X0' in text:
                meta['X0'] = [float(x) for x in text.strip('X0').strip('\t').split('\t')]
        meta['start'] = 0
        meta['YAGtrig'] = 2
    if version==0 or version==1 or version==2:
        with open(filepath, 'r') as f:
            lines=[]
            for i in range(header_lines):
                lines.append(f.readline().strip('\n'))
        meta={}
        for text in lines:
            if 'run number' in text:
                meta['run'] = int(split_strip(text))
            if 'command file' in text:
                meta['command file'] = split_strip(text).split('\\')[-1].strip('.txt')
            if 'cycle number' in text:
                meta['cycle'] = int(split_strip(text))
            if 'comment' in text:
                meta['comment'] = text.split(':')[1]
            if 'wavemeter(cm-1)' in text:
                meta['frequency'] = float(split_strip(text))
            if 'offset' in text:
                meta['cavity_offset'] = float(split_strip(text))
            if 't0' in text:
                H_M_S = text.strip('t0').strip().split('\t')[0].split(' ')[-1].split(':')
                hours_in_sec = float(H_M_S[0])*60*60
                min_in_sec = float(H_M_S[1])*60
                sec = float(H_M_S[2])
                meta['trigtime'] = hours_in_sec+min_in_sec+sec
            if 'delta t' in text:
                meta['dt'] = [float(x) for x in text.strip('delta t\t').split('\t')][0]*1000 #convert seconds to ms
            if 'HFwavemeter' in text:
                meta['frequency_HF']  = float(split_strip(text))
        meta['start'] = 0
        meta['YAGtrig'] = 2 #hard coded for now
    if version==3:
        with open(filepath, 'r') as f:
            lines=[]
            for i in range(header_lines):
                lines.append(f.readline().strip('\n'))
        meta={}
        for text in lines:
            if 'run number' in text:
                meta['run'] = int(split_strip(text))
            if 'cycle number' in text:
                meta['cycle'] = int(split_strip(text))
            if 'command file' in text:
                meta['command file'] = split_strip(text).split('\\')[-1].strip('.txt')
            if 'comment' in text:
                meta['comment'] = text.split(':')[1]
            if 'wavemeter readings' in text:
                meta['frequency'] = [float(value) for value in split_strip(text).split(',')]
            if 't0' in text:
                H_M_S = text.strip('t0').strip().split('\t')[0].split(' ')[-1].split(':')
                hours_in_sec = float(H_M_S[0])*60*60
                min_in_sec = float(H_M_S[1])*60
                sec = float(H_M_S[2])
                meta['trigtime'] = hours_in_sec+min_in_sec+sec
            if 'delta t' in text:
                meta['dt'] = [float(x) for x in text.strip('delta t\t').split('\t')][0]*1000 #convert seconds to ms

        meta['start'] = 0
        meta['YAGtrig'] = 2 #hard coded for now
    if version==4:
        with open(filepath, 'r') as f:
            lines=[]
            for i in range(header_lines):
                lines.append(f.readline())
        meta_code = ''.join(lines[1:-7])
        ldict={}
        exec(meta_code,globals(),ldict)
        meta = ldict['meta_dict']
        for text in lines:
            if 't0' in text:
                H_M_S = text.strip('t0').strip().split('\t')[0].split(' ')[-1].split(':')
                hours_in_sec = float(H_M_S[0])*60*60
                min_in_sec = float(H_M_S[1])*60
                sec = float(H_M_S[2])
                meta['trigtime'] = hours_in_sec+min_in_sec+sec
            if 'delta t' in text:
                meta['dt'] = [float(x) for x in text.strip('delta t\t').split('\t')][0]*1000 #convert seconds to ms
        meta['start'] = 0
        meta['YAGtrig'] = 2 #hard coded for now
    return meta


def import_raw_data_Cleverscope(filepath,usecols=(1,2)):
    '''Imports raw traces from data columns in single cleverscope text file.
    Input is file path.'''
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

    Parameters returned as dict:
    dt: time step of data traces (ms)
    start: start time of data traces, relative to trigger (ms)
    nsample: number of elements in time series
    trigtime: UNIX time of trigger
    '''
    with open(filepath, 'r') as f:
        lines=[]
        for i in range(15):
            lines.append(f.readline())
    params={}
    for text in lines:
        if 'delta' in text:
            dt_ms = np.round(float(text.split('\t')[1].strip())*10**3,decimals=6)
            params['dt']=dt_ms
        elif 'start' in text:
            start_ms = np.round(float(text.split('\t')[1].strip())*10**3,decimals = 6)
            params['start']=start_ms
        elif 'nsample' in text:
            nsample = int(float(text.split('\t')[1].strip()))
            params['nsample']=nsample
        elif 'TriggerTime' in text:
            # The time is given as the number of days since 12/29/1899 11:00:00 UTC
            # We can convert this to UNIX by subtracting 2209208400 seconds
            # Ask Arian if you are curious
            time_sec = float(text.split('\t')[1].strip())*24*3600 #convert days to seconds
            time_unix = time_sec - 2209208400
            params['trigtime']=time_unix
    params['YAGtrig'] = 0
    return params


def smooth(data,window=5,poly_order=3):
    '''Function for smoothing data. Currently uses Savitzky-Golay filter,
    which fits a window of data onto a polynomial of some order, and then
    uses the polynomial to estimate the value'''
    #window value must be odd
    if window % 2 == 0:
        window+=1
    smoothed_data = savgol_filter(data, window, poly_order)
    return smoothed_data

def sliceIntegrate(data,time_ms,start_stop,fig_num=1):
    '''Integrate time slice of the supplied data trace from start to stop.
        start_stop = [start,stop] in ms'''
    start_i = np.searchsorted(time_ms,start_stop[0])
    stop_i = np.searchsorted(time_ms,start_stop[1])
    t = slice(start_i,stop_i)
    dt = np.round(time_ms[1]-time_ms[0],decimals=6)
    integrated = np.round(data[t].sum()*dt,decimals=6)
    plt.figure(fig_num)
    plt.plot(time_ms[t],data[t])
    return integrated

## Other functions ##

def labelBUB(param_array,blocked_bool):
    '''Iterate through a series of files and compiles ODs and parameters.
        Labels the whole series as blocked (True) or unblocked (False).
        Returns an array of ODs and parameters'''
    # not sure what this does. looks broken

    for single_param in param_array:
        single_param['blocked'] = blocked_bool
    return [ODs,params]

'''Extracts trigger timestamps from parameter arrays. Times are in seconds relative to the initial file timestamp'''
def extractTimestamps(params,subtract_offset=True):
    timestamps = []
    if subtract_offset:
        for single_params in params:
            if len(timestamps)==0:
                initial = single_
                timestamps.append(0)
            else:
                elapsed = single_params['trigtime'] - initial
                timestamps.append(elapsed)
    else:
            for single_params in params:
                unix_time = single_params['trigtime']
                timestamps.append(unix_time)
    return np.array(timestamps)


'''Extracts trigger timestamps from 2 (blocked/unblocked) parameter arrays,
    relative to whichever was first'''
def extractTimestampsBUB(blocked_params,unblocked_params):
    blocked_timestamps = []
    unblocked_timestamps = []
    if blocked_params[0]['trigtime'] < unblocked_params[0]['trigtime']:
        blocked_first = True
    else:
        blocked_first = False
    for i in range(len(blocked_params)):
        if blocked_first:
            if len(blocked_timestamps)==0:
                initial = blocked_params[i]['trigtime']
                blocked_timestamps.append(0)
                elapsed_unblocked = unblocked_params[i]['trigtime'] - initial
                unblocked_timestamps.append(elapsed_unblocked)
            else:
                elapsed_blocked = blocked_params[i]['trigtime'] - initial
                elapsed_unblocked = unblocked_params[i]['trigtime'] - initial
                blocked_timestamps.append(elapsed_blocked)
                unblocked_timestamps.append(elapsed_unblocked)
        else:
            if len(unblocked_timestamps)==0:
                initial = unblocked_params[i]['trigtime']
                unblocked_timestamps.append(0)
                elapsed_blocked = blocked_params[i]['trigtime'] - initial
                blocked_timestamps.append(elapsed_blocked)
            else:
                elapsed_blocked = blocked_params[i]['trigtime'] - initial
                elapsed_unblocked = unblocked_params[i]['trigtime'] - initial
                blocked_timestamps.append(elapsed_blocked)
                unblocked_timestamps.append(elapsed_unblocked)
    return [blocked_timestamps,unblocked_timestamps]




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


## File I/O Subfunctions ##

def _write_CSV_rows(data_arrays,filepath):
    '''This function writes arrays in rows to a CSV file. Subfunction for
    write_CSV_rows.

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

def isFloat(value):
    '''Checks if input is float.'''
    # Probably easier to use isinstance
    try:
        float(value)
        return True
    except:
        return False

def gen_filepath_from_parent(path):
    '''Generates path to a file located in a parellel directory
    from the current working directory (cwd). Specifically, if the cwd is in
    parent_dir/cwd, then it generates a path to parent_dir/path
    '''
    current = pathlib.Path.cwd()
    parrent_dir = current.parents[0]
    new_path = parent_dir / path
    return new_path


def find_last_file(folder_path,file_name,initial=1):
    found_end = False
    n = 0
    while not found_end:
        file_path = pathlib.Path(gen_data_filepath(folder_path,n+initial,file_name))
        if file_path.is_file():
            n+=1
        else:
            found_end = True
    return n

def extract_cavity(metadata):
    return np.array([meta['frequency_cavity'] for meta in metadata])

def extract_meter(metadata):
    return np.array([meta['frequency_meter'] for meta in metadata])


def gen_data_filepath(folder_path, num, name, version=1):
    '''Generate file path for single text file named "name_num.txt", located
    in the folder given by fikder_path. Note folder_path should be an absolute
    path
    '''
    if version==3 or version==4:
        version=2
    fmt = {-1: '03d', 0:'04d', 1:'01d',2:'01d'}[version]
    num_str = format(num,fmt)
    folder_path = pathlib.Path(folder_path)
    file =  name + num_str + '.txt'
    full_path = folder_path / file
    return full_path



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


#Duplicate for backwards compatability
def sortData(indep_var,depend_var):
    #print(len(indep_var))
    #print(len(depend_var))
    if len(indep_var) != len(depend_var):
        print('Number of independent and dependent variables do not match')
    else:
        if not isinstance(indep_var,list):
            indep_var = list(indep_var)
        sorted_DV = [DV for IV,DV in sorted(zip(indep_var,depend_var))]
    return np.array(sorted_DV)

def sort_data(indep_var,depend_var,return_IV=True):
    #print(len(indep_var))
    #print(len(depend_var))
    if len(indep_var) != len(depend_var):
        print('Number of independent and dependent variables do not match')
    else:
        if not isinstance(indep_var,list):
            indep_var = list(indep_var)
        sorted_DV = [DV for IV,DV in sorted(zip(indep_var,depend_var))]
        if return_IV:
            sorted_IV = sorted(indep_var)
            return [np.array(sorted_IV),np.array(sorted_DV)]
        else:
            return np.array(sorted_DV)

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
