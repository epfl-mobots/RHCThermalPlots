'''
Author(s):
    Cyril Monette, EPFL, cyril.monette@epfl.ch
'''

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from thermalframe import ThermalFrame, NoValidSensors
from RHCImaging.VideoManagment.videolib import generateVideoFromList, fig_to_rgb_array, cropFrameToContent

def extractAmbientTemp(temps:pd.DataFrame, num_sensors:int = 3)->pd.Series:
    '''
    Extracts the ambient temperature from the thermal dataframe.
    The ambient temperature is estimated as the average of the lowest num_sensors sensors values.

    Parameters:
    - temps: pd.DataFrame containing the temperatures (index: datetime)
    - num_sensors: number of sensors to consider for the ambient temperature estimation (default: 3)
    Returns:
    - ambient_temp: pd.Series containing the estimated ambient temperature (index: datetime)
    '''
    ambient_temp = temps.apply(lambda row: np.nanmean(np.sort(row.values)[:num_sensors]), axis=1)
    return ambient_temp

def _lab_update_isotherm(hive_min_temp:float, hive_max_temp:float) -> float:
    # When hut/min temp is 11.2175C, tracking iso = 15C
    # Note: see Science Robotics S.M. for explanation of the values
    if hive_min_temp < 11.2175:
        return 15.0  # minimum tracking temperature will be 15C
    else:
        # NOTE: LAB TESTING!
        _isoth = round(hive_min_temp + 3.0, 1)
        # Clip value between 15 and hive_max_temp
        _isoth = np.maximum(_isoth, 15)  # Ensure a minimum isotherm of 15°C
        _isoth = np.minimum(_isoth, hive_max_temp - 1)  # Ensure a maximum isotherm of hive_max_temp - 1°C
        return _isoth

def _hut_update_isotherm(hive_min_temp:float, hive_max_temp:float) -> float:
    # When hut/min temp is 11.2175C, tracking iso = 15C
    # Note: see Science Robotics S.M. for explanation of the values
    #_isoth = 0.92 * hive_min_temp + 4.68 # original formula
    _isoth = 0.92 * hive_min_temp + 8
    _isoth = np.maximum(_isoth, 15)  # Ensure a minimum isotherm of 15°C
    _isoth = np.minimum(_isoth, hive_max_temp - 1)  # Ensure a maximum isotherm of hive_max_temp - 1°C
    return round(_isoth, 1)

def update_isotherm(hive_min_temp:float, hive_max_temp:float) -> float:
    '''wrapper'''
    #return(self._lab_update_isotherm(hive_min_temp, hive_max_temp))
    return(_hut_update_isotherm(hive_min_temp, hive_max_temp))

def generateThermalDF(df:pd.DataFrame)->pd.DataFrame:
    '''
    Generates a panda df for temperatures that is friendly with the thermalutil.py library.
    This means the columns are t00-t64 and one line is one timestamp.
    Parameters:
    - df: pd.DataFrame containing the temperatures in an influxdb format.

    returns:
    - upper: pd.DataFrame containing the upper hive temperatures
    - lower: pd.DataFrame containing the lower hive temperatures
    '''
    _index = df.index.unique()
    upper = pd.DataFrame(index=_index)
    lower = pd.DataFrame(index=_index)
    df_upper = df[(df['_measurement'] == 'tmp') & (df['inhive_loc'] == 'upper')]
    df_lower = df[(df['_measurement'] == 'tmp') & (df['inhive_loc'] == 'lower')]

    for i in range(64):
        column_name = f't{i:02d}'
        # For every index of thermal_df, get the temperature which has the same datetime in df
        upper[column_name] = df_upper[df_upper['_field'] == column_name]['_value']
        lower[column_name] = df_lower[df_lower['_field'] == column_name]['_value']

    # Set the right column names
    upper.columns = [f't{i:02d}' for i in range(64)]
    lower.columns = [f't{i:02d}' for i in range(64)]

    # Convert every column to float type
    for col in upper.columns:
        upper[col] = upper[col].astype(float)
    for col in lower.columns:
        lower[col] = lower[col].astype(float)

    # Suppress eratic values
    upper[upper < -50] = np.nan
    lower[lower < -50] = np.nan
    return upper, lower

def readFromFile(filepath:str, verbose:bool=False)->pd.DataFrame:
    '''
    Reads a .dat or .csv file and returns a pandas dataframe with the thermal data.
    The file is expected to have the following format:
    - First column: datetime in ISO format
    - Next 64 columns: temperature data from sensors t00 to t63
    - Optional next column: validity flag (True/False)
    - Optional next 10 columns: target temperatures
    - Optional next 10 columns: h_avg_temps
    - Optional next 10 columns: pwm values
    
    Parameters:
    - filepath: path to the .dat file
    
    Returns:
    - df_therm_data: pandas dataframe containing thermal data (datetime as index, 64 columns for sensor data and potentially one more column for validity)
    '''

    assert filepath.endswith('.dat') or filepath.endswith('.csv'), "File must be a .dat or .csv file"

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)

    else:
        # First skip the lines that do not start with "20" (the millenia)
        skip = 0
        with open(filepath, 'r') as f:
            lines = f.readlines()
            while skip < len(lines) and not lines[skip].startswith('20'):
                skip += 1

        try:
            # Assuming data is in a structured format like CSV or similar
            df = pd.read_csv(filepath, delimiter=',', skiprows=skip, header=None)
        except FileNotFoundError:
            print(f"File '{filepath}' not found.")

    validity_flag = False
    # make the columns as being : datetime, 64 columns for sensor data
    temps_labels = [f't{i:02d}' for i in range(64)]
    if df.shape[1] == 1+ 64 + 1 + 10 + 10 + 10 or df.shape[1] == 1 + 64 + 1: # datetime + 64 temps + 1 validity + 10 targets + 10 h_avg_temps + 10 pwm
        validity_flag = True
        if verbose:
            print("File has validity flag")
        temps_labels.append('validity')
    other_cols = df.shape[1] - (1 + 64 + (1 if validity_flag else 0))
    other_cols_labels = []
    if other_cols > 0:
        other_cols_labels = ['target' for _ in range(10)] + ['h_avg_temps' for _ in range(10)] + ['pwm' for _ in range(10)]
    df.columns = ['datetime'] + temps_labels + other_cols_labels
    df.set_index('datetime', inplace=True)
    if verbose:
        print(f"dataframe preview:\n {df.head()}")

    temps = df.iloc[:,0:64]
    if validity_flag:
        #typecast the validity column to boolean
        df.loc[:,'validity'] = df.loc[:,'validity'].astype(bool)
        temps = temps[df.loc[:,'validity'] == True]
    # Delete the validity column if it exists
    if 'validity' in df.columns:
        df.drop(columns=['validity'], inplace=True)
    if verbose:
        print(f"temps preview:\n {temps.head()}")

    if df.shape[1] > 64: # 64 temps
        targets = df.iloc[:,64:74]
        targets.columns = [f'h{i:02d}' for i in range(targets.shape[1])]
    else:
        targets = pd.DataFrame(index=temps.index) # Empty dataframe
    if verbose:
        print(f"targets preview:\n {targets.head()}")

    return temps, targets

def preview(df_therm_data, 
            show_sensors:bool=False, 
            contours:list[float] = None, 
            vmin=None,
            vmax=None,
            rows=None):
    '''
    Preview the thermal data in the file through 6 plots of equally spaced time points.
    param df_therm_data: pandas dataframe containing thermal data (datetime as index, 64 columns for sensor data and potentially one more column for validity)
    param rows: list of row indices to preview
    '''
    # check that the df is valid
    if not isinstance(df_therm_data, pd.DataFrame):
        raise ValueError("The input is not a pandas dataframe")
    if df_therm_data.shape[1] != 65 and df_therm_data.shape[1] != 64:
        raise ValueError(f"The input dataframe does not have the correct shape (needs to be 64 or 65 columns and got {df_therm_data.shape[1]})")
    if df_therm_data.index.name != 'datetime':
        raise ValueError("The input dataframe does not have the correct index name (needs to be 'datetime')")
    if df_therm_data.iloc[:,0].name !='t00':
        raise ValueError("The input dataframe does not have the correct column names (first column should be 't00')")
    
    # iterate through a few rows
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 7))
    n_data_points = df_therm_data.shape[0]
    if rows is None:
        rows = np.linspace(0, n_data_points-1, 6, dtype=int)

    print("=====================================================================================================================================================")
    print(f"Previewing thermal data:")
    for i, row in enumerate(rows):
        temps = df_therm_data.iloc[row]
        temps_np = temps.to_numpy()
        try:
            tf = ThermalFrame(temperature_data=temps_np)

        except NoValidSensors as e:
            print(f"Skipping row {row} due to no valid sensors")
            continue
        
        tf.calculate_thermal_field(verbose=False)

        x, y = tf.get_max_temp_pos()

        print(f"Max temp at {x}, {y} for row {row} [tmax = {tf.max_temp} °C]")
        a = ax.flat[i]
        tf.plot_thermal_field(a,show_cb=True, show_sensors=show_sensors, show_max_temp=True, contours=contours, annotate_contours=True, v_min=vmin,v_max=vmax)
        a.set_title(f"Row {row} -> {str(df_therm_data.iloc[row].name)}") # Overwrite the title

def generateThermalVideo(df_therm_data:pd.DataFrame, 
                         video_name:str, 
                         dest:str='outputVideos',
                         fps:int=10,
                         hive_nb:int=-1,
                         show_cb:bool=False,
                         show_sensors:bool=False, 
                         show_max_temp:bool=False,
                         contours:list[float]=[], 
                         vmin:float=None, vmax:float=None,
                         faulty_s:list[int]=[],
                         padding:int=50,
                         verbose:bool=False):
    '''
    Generates a video from the thermal data in the dataframe.
    The smallest gap between two timestamps is used to determine the expected time resolution.
    '''

    time_diffs = df_therm_data.index.to_series().diff().dropna()
    time_res = time_diffs.min()
    if verbose:
        print(f"Time resolution of the data: {time_res}")
    time_index = pd.date_range(start=df_therm_data.index.min(), end=df_therm_data.index.max(), freq=time_res)

    # Put NaN values for faulty sensors
    for s in faulty_s:
        col_name = f't{s:02d}'
        if col_name in df_therm_data.columns:
            df_therm_data.loc[:,col_name] = np.nan
        else:
            print(f"Warning: sensor {s} is not in the dataframe columns")
    
    # Set the vmin and vmax if not provided
    if vmin is None:
        vmin = df_therm_data.min().min()
    if vmax is None:
        vmax = df_therm_data.max().max()

    if contours == []:
        contours = list(np.arange(np.floor(vmin), np.ceil(vmax), 1))

    # Find typical frame size:
    fig, ax = plt.subplots(figsize=(13, 7))
    _tf = ThermalFrame(temperature_data=df_therm_data.iloc[0].to_numpy(), hive_id=hive_nb)
    _tf.plot_thermal_field(ax, show_cb=show_cb, show_sensors=show_sensors, contours=contours, v_min=vmin, v_max=vmax)
    example_frame = fig_to_rgb_array(fig)
    example_frame = cropFrameToContent(example_frame, padding=padding)
    height, width, channels = example_frame.shape
    if verbose:
        print(f"Video frame size: {height}x{width}x{channels}")
    plt.close(fig) # We close the figure to avoid displaying it

    # Generate the frames:
    frames = []
    for t in tqdm(time_index, desc="Generating frames"):
        if t not in df_therm_data.index:
            if verbose:
                print(f"Time {t} is not in the dataframe index")
            # Append black frame
            frames.append(np.zeros((height, width, channels), dtype=np.uint8))
            continue
        try:
            tf = ThermalFrame(temperature_data=df_therm_data.loc[t].to_numpy(), hive_id=hive_nb, ts=t, bad_sensors=faulty_s)
        except NoValidSensors as e:
            print(f"Skipping time {t} due to no valid sensors")
            # Append black frame
            frames.append(np.zeros((height, width, channels), dtype=np.uint8))
            continue
        tf.calculate_thermal_field(verbose=False)
        fig, ax = plt.subplots(figsize=(13, 7))
        tf.plot_thermal_field(ax, show_cb=show_cb, show_sensors=show_sensors, show_max_temp=show_max_temp, contours=contours, v_min=vmin, v_max=vmax)
        ax.set_title(f"Thermal field for hive {tf.hive_id} at {str(t)}") # Overwrite the title
        _frame = fig_to_rgb_array(fig)
        _frame = cropFrameToContent(_frame, padding=padding)
        frames.append(_frame)
        plt.close(fig) # We close the figure to avoid displaying it
    if verbose:
        print(f"Generated {len(frames)} frames for the video")

    # Generate the video
    generateVideoFromList(frames, dest=dest, name=video_name, fps=fps, grayscale=False)


def trend_filter(data, lmbd=50, order=2):
    ''' Trend filtering
        type: 'L1' or 'HP'

        Refs:
        - https://towardsdatascience.com/introduction-to-trend-filtering-with-applications-in-python-d69a58d23b2
    '''
    import cvxpy as cp
    _data = None
    if isinstance(data, list):
        _data = np.array(data)
    elif isinstance(data, np.ndarray):
        _data = data
    else:
        print("Data seems to be in the wrong format!!!")

    # Regularization parameter
    # Convergence to original data as λ→0
    # Finite convergence to best affine fit as λ→∞
    vlambda = lmbd
    n = _data.size

    # solver = cp.ECOS
    solver = cp.CVXOPT
    reg_norm = order    # 1 = L1 trend filtering
                        # 2 = H-P filter

    # Form second difference matrix.
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)

    # Solve l1 trend filtering problem.
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(data - x) + vlambda * cp.norm(D @ x, reg_norm))
    prob = cp.Problem(obj)

    # ECOS and SCS solvers fail to converge before
    # the iteration limit. Use CVXOPT instead.
    prob.solve(solver=solver, verbose=True)
    print('Solver status: {}'.format(prob.status))

    # Check for error.
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    else:
        print("optimal objective value: {}".format(obj.value))
        return x.value