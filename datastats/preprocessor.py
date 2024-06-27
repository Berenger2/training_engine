import pandas as pd
import numpy as np

from scipy.fft import fft
from scipy.stats import skew, kurtosis
import statistics as stc



import configparser

config = configparser.ConfigParser()
config.read('config.ini')
FRAME_SIZE = config.getint('Default', 'FRAME_SIZE')


def get_mean_acceleration(signal):
    mean = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_mean = np.sum(signal[i:i+FRAME_SIZE])/FRAME_SIZE
            mean.append(current_mean)
    return mean

def get_std(signal):
    fin_std = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:        
            current_std = np.std(signal[i:i+FRAME_SIZE])
            fin_std.append(current_std)
    return fin_std

def get_variance(signal):
    fin_var = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_var = np.var(signal[i:i+FRAME_SIZE])
            fin_var.append(current_var)
    return fin_var

def get_rms_acceleration(signal):
    rms = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_rms = np.sqrt(np.sum(signal[i:i+FRAME_SIZE]**2)/FRAME_SIZE)
            rms.append(current_rms)
    return rms

def get_peak_acceleration(signal):
    peak = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_frame = max(signal[i:i+FRAME_SIZE])
            peak.append(current_frame)
    return np.array(peak)

from scipy.stats import skew

def get_skewness(signal):
    fin_skew = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_skew = skew(signal[i:i+FRAME_SIZE])
            fin_skew.append(current_skew)
    return fin_skew

from scipy.stats import kurtosis

def get_kurtosis(signal):
    fin_kurt = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_kurt = kurtosis(signal[i:i+FRAME_SIZE])
            fin_kurt.append(current_kurt)
    return fin_kurt

def get_crest_factor(signal):
    crest_fac = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            curr_crest_fac = np.max(np.abs(signal[i:i+FRAME_SIZE])) / skew(signal[i:i+FRAME_SIZE])
            crest_fac.append(curr_crest_fac)                             
    return crest_fac

def get_margin_factor(signal):
    mar_fac = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            curr_mar_fac = np.max(np.abs(signal[i:i+FRAME_SIZE])) / ((np.sum(np.sqrt(np.abs(signal[i:i+FRAME_SIZE])))/ FRAME_SIZE**2))
            mar_fac.append(curr_mar_fac)                             
    return mar_fac

def get_shape_factor(signal):
    fin_shape_fact = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            cur_shape_fact = np.sqrt(((np.sum(signal[i:i+FRAME_SIZE]**2))/FRAME_SIZE) / (np.sum(np.abs(signal[i:i+FRAME_SIZE]))/FRAME_SIZE))
            fin_shape_fact.append(cur_shape_fact)

    return fin_shape_fact

def get_impulse_factor(signal):
    impulse_factor = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_impls = max(np.abs(signal[i:i+FRAME_SIZE]))/(np.sum(np.abs(signal[i:i+FRAME_SIZE])/FRAME_SIZE))
            impulse_factor.append(current_impls)
    return impulse_factor

def get_A_factor(signal):
    A_factor = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_factor = max(signal[i:i+FRAME_SIZE])/(np.std(signal[i:i+FRAME_SIZE])*np.var(signal[i:i+FRAME_SIZE]))
            A_factor.append(current_factor)
    return A_factor

def get_B_factor(signal):
    B_factor = []
    for i in range(0, len(signal), FRAME_SIZE):
        if len(signal[i:i+FRAME_SIZE]) > 4000:
            current_b_factor = (kurtosis(signal[i:i+FRAME_SIZE]))*(np.max(np.abs(signal[i:i+FRAME_SIZE])) / skew(signal[i:i+FRAME_SIZE]))/(np.sqrt((np.sum((signal[i:i+FRAME_SIZE] - (np.sum(signal[i:i+FRAME_SIZE])/FRAME_SIZE))**2))/(FRAME_SIZE-1)))
            B_factor.append(current_b_factor)
    return B_factor

list_features_function = [get_peak_acceleration, get_rms_acceleration, get_crest_factor,get_std, get_variance,
                          get_skewness, get_kurtosis, get_shape_factor, get_impulse_factor, get_margin_factor,
                         get_mean_acceleration, get_A_factor, get_B_factor]

print('Number of feature extruction methods: ', len(list_features_function))

def get_all_fetures(signal):
    stationary_features = []
    for func in list_features_function:
        f = func(signal)
        stationary_features.append(f)
    return stationary_features



def create_df(features_b, features_t):
    ready_data=pd.DataFrame({'Mean_b': np.array(features_b).T[:, 0],
                         'STD_b': np.array(features_b).T[:, 1],
                         'Variance_b': np.array(features_b).T[:, 2],
                         'RMS_b': np.array(features_b).T[:, 3],
                         'Peak_val_b': np.array(features_b).T[:, 4],
                         'Skewness_b': np.array(features_b).T[:, 5],
                         'Kurtosis_b': np.array(features_b).T[:, 6],
                         'Crest_factor_b': np.array(features_b).T[:, 7],
                         'Margin_factor_b': np.array(features_b).T[:, 8],
                         'SHape_factor_b': np.array(features_b).T[:, 9],
                         'Impulse_factor_b': np.array(features_b).T[:, 10],
                         'A_factor_b': np.array(features_b).T[:, 11],
                         'B_factor_b': np.array(features_b).T[:, 12],
                         'Mean_t': np.array(features_t).T[:, 0],
                         'STD_t': np.array(features_t).T[:, 1],
                         'Variance_t': np.array(features_t).T[:, 2],
                         'RMS_t': np.array(features_t).T[:, 3],
                         'Peak_val_t': np.array(features_t).T[:, 4],
                         'Skewness_t': np.array(features_t).T[:, 5],
                         'Kurtosis_t': np.array(features_t).T[:, 6],
                         'Crest_factor_t': np.array(features_t).T[:, 7],
                         'Margin_factor_t': np.array(features_t).T[:, 8],
                         'SHape_factor_t': np.array(features_t).T[:, 9],
                         'Impulse_factor_t': np.array(features_t).T[:, 10],
                         'A_factor_t': np.array(features_t).T[:, 11],
                         'B_factor_t': np.array(features_t).T[:, 12]
                        })
    return ready_data



def dataTransformer(signalb: object, signalt: object) -> object:
    mask_b = signalb.isnull()
    mask_t = signalt.isnull()
    signalb = signalb[~mask_b]
    signalt = signalb[~mask_t]
    features_b = get_all_fetures(signalb)
    features_t = get_all_fetures(signalt)
    data = create_df(features_b, features_t)
    return data