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
plt.style.use('seaborn-whitegrid')
import seaborn as sns
import scipy 
from scipy.stats import kurtosis, skew

# metrics
import tensorflow as tf
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
    print(path_to_read_final_data)
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

def plot_history(log_key, object):
    fig, axs = plt.subplots(1,5, figsize=(20, 4), tight_layout=True)

    axs[0].plot(object.history_log[log_key].history['loss'], label='loss')
    axs[0].plot(object.history_log[log_key].history['val_loss'], label='val_loss')
    axs[0].set_title('loss')

    axs[1].plot(object.history_log[log_key].history['RMSLETF'], label='RMSLETF')
    axs[1].plot(object.history_log[log_key].history['val_RMSLETF'], label='val_RMSLETF')
    axs[1].set_title('RMSLETF')

    axs[2].plot(object.history_log[log_key].history['mae'], label='mae')
    axs[2].plot(object.history_log[log_key].history['val_mae'], label='val_mae')
    axs[2].set_title('mae')

    axs[3].plot(object.history_log[log_key].history['root_mean_squared_error'], label='root_mean_squared_error')
    axs[3].plot(object.history_log[log_key].history['val_root_mean_squared_error'], label='val_root_mean_squared_error')
    axs[3].set_title('root_mean_squared_error')

    axs[4].plot(object.history_log[log_key].history['mean_absolute_percentage_error'], label='mean_absolute_percentage_error')
    axs[4].plot(object.history_log[log_key].history['val_mean_absolute_percentage_error'], label='val_mean_absolute_percentage_error')
    axs[4].set_title('mean_absolute_percentage_error') 

    plt.suptitle(f'{log_key}')
    plt.show()

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

def RMSLETF(y_pred:tf.Tensor, y_true:tf.Tensor) -> tf.float64:
    """
    The Root Mean Squared Log Error (RMSLE) metric for TensorFlow / Keras
     
    :param y_true: The ground truth labels given in the dataset
    :param y_pred: Predicted values
    :return: The RMSLE score
    """
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64) 
    y_pred = tf.nn.relu(y_pred) 
    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(tf.math.log1p(y_pred+1.), tf.math.log1p(y_true+1.))))

metrics = {'RMSLE': RMSLE, 'MAE': mean_absolute_error, 'RMSE': RMSE, 'MAPE': mean_absolute_percentage_error}

def plot_evaluation_evolution_through_steps(evaluation, ax=None):
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

def plot_all_predictions(y_fit_train, y_fit_test, df, model_name='',window_size=14 ):    
    palette = dict(palette='husl', n_colors=64)
    # fig, ax1 = plt.subplots(1, 1, figsize=(11, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # train data
    ax1 = df['Infected'][y_fit_train.index].plot(color='grey', zorder=0, ax=ax1)
    y_fit_train['y_step_1'].plot(ax=ax1, color='k', style='.-', legend='First Forecast')
    plot_multistep(y_fit_train, ax=ax1, palette_kwargs=palette)
    _ = ax1.legend(['Actual series', 'First step forecast', 'Forecast'])

    # test data
    ax2 = df['Infected'][y_fit_test.index].plot(color='grey', zorder=0, ax=ax2)
    y_fit_test['y_step_1'].plot(ax=ax2, color='k', style='.-', legend='First Forecast')
    plot_multistep(y_fit_test, ax=ax2, palette_kwargs=palette)
    _ = ax2.legend(['Actual series', 'First step forecast', 'Forecast'])

    # set title
    ax1.set_title('Train data')
    ax2.set_title('Test data')
    fig.suptitle(f'{model_name} model for window size {window_size} and steps ahead 14')

    # label axes
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Infected')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Infected')

    # set same y scale for both axes
    ax2.set_ylim(ax1.get_ylim())

    plt.show()