# '''Widget wrapper for getRawDataset'''
# def wrapper_getRawDataset(w):
#         print('Generating raw data from ',folder_path_w.value)
#         #results = processData(folder_path_w.value,start_num_w.value,stop_num_w.value,start_setpnt_w.value,stop_setpnt_w.value,setpnt_int_w.value,print_w.value)
#         results = getRawDataset(folder_path_w.value,start_num_w.value,stop_num_w.value,print_w.value)
#         w.value = results
#         print('Done')


# '''Widget wrapper, not used'''
# def wrapper_calcODs(w):
#         print('Calculating ODs from ',folder_path_w.value)
#         #results = processData(folder_path_w.value,start_num_w.value,stop_num_w.value,start_setpnt_w.value,stop_setpnt_w.value,setpnt_int_w.value,print_w.value)
#         results = calculateODs(*rawdata_lb.value)
#         w.value = results
#         print('Done')
#         return



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
