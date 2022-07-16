# data handling
import pandas as pd
import numpy as np
from datetime import date

# path handling
import os
from pathlib import Path
import matplotlib.pyplot as plt

# warnings config
import warnings
warnings.filterwarnings('ignore')

#======================= CONSTANTS =================

# today's date
today_date = date.today().strftime("%d%b%Y")

# name of the files
file_name = f'spanish-covid-19-data-renave-{today_date}.csv'

# data path
base_dir = Path(os.getcwd())

# path to save last raw data
path_to_save_raw_data = base_dir / 'data' / 'raw-data' / file_name	

# path to save preprocessed data
path_to_save_processed_data = base_dir / 'data' / 'processed-data' / file_name	

# path to save final data
path_to_save_final_data = base_dir / 'data' / 'final-data' / file_name	

# data url
data_url = 'https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv'

# ======================= FUNCTIONS =================
def data_retrieval_from_url():
    # get data from url
    df = pd.read_csv(data_url, 
                parse_dates=['fecha'], encoding='latin-1', keep_default_na=False)
    # # save raw data to disk
    # if not os.path.exists(path_to_save_raw_data):
    #     os.mkdir(base_dir / 'data' / 'raw-data' )
    df.to_csv(path_to_save_raw_data, index=False)
    print(f'\n\tUpdated raw data saved in {path_to_save_raw_data}')
    return df 

def preprocess_data(df, recovery_period=None, smooth=True):
    df = df.drop(columns=['num_hosp', 'num_uci'])

    df = df.rename(columns={'provincia_iso':'ISO3',
            'fecha':'Date', 
            'num_casos':'Infected', 
            'num_def':'Deaths'})
    # group by date to delete province data
    df = df.groupby(['Date']).sum()

    # add population
    df['Population'] = 46796540

    # select data for SI (Susceptible, Infected)
    df = df[['Population', 'Infected', 'Deaths']]

    if smooth is True:
        # smooth data
        df['Infected'] = df['Infected'].rolling(7).mean().dropna().astype(int)
        df['Deaths'] = df['Deaths'].rolling(7).mean().dropna().astype(int)

    # add recovered
    if recovery_period is None:
        recovery_period = 10
    df['Recovered'] = 0
    df['Recovered'].iloc[recovery_period:] = df['Infected'].iloc[:-recovery_period] - df['Deaths'].iloc[:-recovery_period]

    # ensure not nan values
    df.dropna(inplace=True)

    # select positive data
    df = df[df.index > pd.to_datetime('2020-03-01')]

    return df


if __name__ == '__main__':
    # read raw data
    df = data_retrieval_from_url()

    # preprocess raw data (without smoothing)
    preprocessed_df = preprocess_data(df, smooth=False)
    # save processed data to disk
    preprocessed_df.to_csv(path_to_save_processed_data, index=True)
    print(f'\n\tUpdated processed data saved in {path_to_save_processed_data}')

    # preprocess raw data (with smoothing)
    final_df = preprocess_data(df, smooth=True)
    # save processed data to disk
    final_df.to_csv(path_to_save_final_data, index=True)
    print(f'\n\tUpdated final data saved in {path_to_save_final_data}')
    print()
