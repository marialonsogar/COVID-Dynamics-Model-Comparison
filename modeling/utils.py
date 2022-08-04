#=======================SETUP========================
# data handling
import pandas as pd
import numpy as np

# path handling
from pathlib import Path
import os
from glob import glob

# analysis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
from scipy.stats import kurtosis, skew

 

# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error

#=======================END==============================

#=======================FUNCTIONS========================

# data loading
def load_last_final_data(file_name:str=None, from_folder:str='final'):
    """
    Loads the last final data from folder, or the specified file

    Args:
        file_name: name of the file to load
        from_folder: folder to load from 
    
    Returns:
        data: DataFrame with the data
    """

    # path of directory with data
    base_dir = Path(os.getcwd()).parents[0]

    # if file_name is not provided, load the last file in the folder
    if file_name is None:
        # name of the directory
        path_to_read_last_final_data = base_dir / 'data' / f'{from_folder}-data' / '*'
        list_of_files = glob(str(path_to_read_last_final_data))
        list_of_valid_files = [file for file in list_of_files if file.endswith('2022.csv')]
        path_to_read_final_data = max(list_of_valid_files, key=os.path.getctime)
        
    # if file_name is provided, load the file
    else:
        # path to read final data
        path_to_read_final_data = base_dir / 'data' / f'{from_folder}-data' / file_name

    # load data
    if from_folder == 'raw':
        column_to_parse_dates = ['fecha']
    else:
        column_to_parse_dates = ['Date']
    df = pd.read_csv(path_to_read_final_data, parse_dates=column_to_parse_dates)
    df.set_index(column_to_parse_dates, inplace=True)
    df = df[(df.index > pd.to_datetime('2020-02-20')) & (df.index < pd.to_datetime('2021-12-01'))]
    # df = df[(df.index > pd.to_datetime('2020-03-20')) & (df.index < pd.to_datetime('2021-12-01'))]
    return df

# analysis
from warnings import simplefilter

simplefilter("ignore")

# # Set Matplotlib defaults
# plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True, figsize=(11, 4))
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=16,
#     titlepad=10,
# )

plot_params = dict(
    color="grey",
    style=".-",
    markeredgecolor="grey",
    markerfacecolor="grey",
)
# %config InlineBackend.figure_format = 'retina'


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax


def plot_acf_pacf(series, title=None, seasonal=True, method=None, **kwargs):
    """Plot acf and pacf of given timeseries"""
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 6)
    plot_acf(series, ax=ax[0], **kwargs)
    plot_pacf(series, ax=ax[1], method=method, **kwargs)

    if seasonal:
        for i in range(1,5):
            ax[0].axvline(7*i, color='lightgray', linestyle='--')
            ax[1].axvline(7*i, color='lightgray', linestyle='--')
    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()

def standarize_residuals(residuals):
    return (residuals - residuals.mean()) / residuals.std()

def diagnostic_checking_residuals(residuals, standarized_residuals=True):
    """
    Diagnostic checking of the residuals
    """
    if standarized_residuals:
        residuals = standarize_residuals(residuals)
        standarized_flag = 'Standarized '
    else:
        standarized_flag = ''

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # line plot of residuals or standardized residuals
    axs[0,0].plot(residuals)
    axs[0,0].axhline(0, color='b', linestyle='--')
    axs[0,0].axhline(-3, color='r', linestyle='--')
    axs[0,0].axhline(3, color='r', linestyle='--')

    axs[0,0].set_title(f'{standarized_flag}Residuals')
    # histogram plus estimated density of standarized residuals, along with a N(0,1) density plotted for reference
    sns.distplot(residuals, ax=axs[0,1])
    x_axis=np.arange(-4,4,0.001)
    axs[0,1].plot(x_axis, 
        scipy.stats.norm.pdf(x_axis, 0,1), 'r', label='N(0,1)')
    axs[0,1].set_title(f'Histogram of {standarized_flag}residuals')
    axs[0,1].legend()
    # normal q-q plot, with Normal reference line
    qqplot(residuals, line='s', ax=axs[1,0])
    axs[1,0].set_title('Normal Q-Q plot')
    # correlogram of the residuals to explore its randomness
    plot_acf(residuals, lags=40, bartlett_confint=False, ax=axs[1,1])
    axs[1,1].set_title(f'Correlogram of {standarized_flag} residuals')

    plt.show()

def residuals_stats(residuals):
    """
    Calculate the statistics of the residuals
    """
    residuals_stats = pd.DataFrame(residuals).describe().T
    residuals_stats['kurtosis'] = kurtosis(residuals) # excess kurtosis of normal distribution (if normal, kurtosis is zero)
    residuals_stats['skewness'] = skew(residuals) # skewness of normal distribution (if normal, skewness is zero)
    return residuals_stats

# tests to check stationarity
def adf_test(timeseries, print_only_result=True):
    """"Augmented Dickey-Fuller test""" 
    # run test
    results = adfuller(timeseries)
    print('\nADF Test Results')
    print('-----------------')
    # display all results
    if print_only_result is False:
        print(f'ADF Statistic: {results[0]}')
        print(f'p-value: {results[1]}')
        print(f'num lags: {results[2]}')
        print('Critical Values:')
        for key, value in results[4].items():
            print(f'\t{key} : {value}')
    # conclusion
    print(f'Result: The timeseries is {"not " if results[1] > 0.05 else ""}stationary')

def kpss_test(timeseries, print_only_result=True):
    """KPSS test"""
    # run test
    statistic, p_value, n_lags, critical_values = kpss(timeseries)
    print('\nKPSS Test Results')
    print('-----------------')
    # display all results
    if print_only_result is False:
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critical Values:')
        for key, value in critical_values.items():
            print(f'\t{key} : {value}')
    # conclusion:
    print(f'Result: The timeseries is {"not " if p_value < 0.05 else ""}stationary')


# =======================================================
# evaluation related functions
def RMSLE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
    """
        Root Mean Squared Log Error (RMSLE) metric 

        Args:        
            y_true: real values
            y_pred: predicted values
            
        Returns:   
            RMSLE score
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def RMSE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
    """
        Root Mean Squared Error (RMSE) metric

        Args:
            y_true: real values
            y_pred: predicted values
            
        Returns:
            RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

metrics = {'RMSLE': RMSLE, 'MAE': mean_absolute_error, 'RMSE': RMSE, 'MAPE': mean_absolute_percentage_error}

def plot_evaluation_evolution_through_steps(evaluation):
    """ 
    Plot the evaluation evolution through steps. 
    
    Args:
        evaluation: dict containing evaluation metrics 
        per step and per baseline step
    """
    # metric evolution subplots
    fig, ax = plt.subplots(4, figsize=(8, 10), tight_layout=True)
    baseline_colors = ['lightsteelblue', 'peachpuff', 'palegreen', 'salmon']
    pd.DataFrame(evaluation).T.loc['baseline_step_1'::2,:].plot(ax=ax, subplots=True, color=baseline_colors)
    pd.DataFrame(evaluation).T.loc[::2,:].plot(ax=ax, subplots=True)
    plt.show()