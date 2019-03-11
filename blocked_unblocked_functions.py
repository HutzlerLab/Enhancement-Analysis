


'''Container for old functions related to blocked unblocked analysis'''
'''Depreciated
    Convert many raw text files to data arrays.
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
