# from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# import tensorflow as tf
from tensorflow import keras
import joblib

import pandas as pd
import numpy as np

def synchronize(dfx,dfy,lead_time=0,lead_freq='D'):
    '''
    synchronizes on index dfx and dfy and return tuple of synchronized data frames
    Note: assumes dfy has only one column
    '''
    # if isinstance(dfx.columns, pd.MultiIndex):
    #     pass
    # else:
    if lead_time > 0:
        dfy.index = dfy.index.shift(-lead_time,freq=lead_freq)
    dfsync=pd.concat([dfx,dfy],axis=1).dropna()
    return dfsync.iloc[:,:-len(dfy.columns)],dfsync.iloc[:,-len(dfy.columns):]


def create_antecedent_inputs(df,ndays=8,window_size=11,nwindows=10):
    '''
    create data frame for CALSIM ANN input
    Each column of the input dataframe is appended by :-
    * input from each day going back to 7 days (current day + 7 days) = 8 new columns for each input
    * 11 day average input for 10 non-overlapping 11 day periods, starting from the 8th day = 10 new columns for each input

    Returns
    -------
    A dataframe with input columns = (8 daily shifted and 10 average shifted) for each input column

    '''
    arr1=[df.shift(n) for n in range(ndays)]
    dfr=df.rolling(str(window_size)+'D',min_periods=window_size).mean()
    arr2=[dfr.shift(periods=(window_size*n+ndays),freq='D') for n in range(nwindows)]
    df_x=pd.concat(arr1+arr2,axis=1).dropna()# nsamples, nfeatures
    return df_x

def split(df, calib_slice, valid_slice):
    if type(calib_slice) == list:
        calib_set = pd.concat([df[slc] for slc in calib_slice],axis=0)
    else:
        calib_set = df[calib_slice]
    if type(valid_slice) == list:
        valid_set = pd.concat([df[slc] for slc in valid_slice],axis=0)
    else:
        valid_set = df[valid_slice]
    return calib_set, valid_set

class myscaler():
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = -float('inf')
    
    def fit_transform(self, data, name=''):
        if isinstance(data, list):
            df = pd.concat(data,axis=0)
            df[df==-2]=float('nan')
            self.min_val = df.min()
            self.max_val = df.max()
        else:
            # if incoming data is not a list of 2D dataframes, then it must be a dataframe with 3 dimensions
            assert isinstance(data.columns, pd.MultiIndex),'Does not support %s data type' % name
            self.min_val = data.min()
            self.max_val = data.max()
            
    def update(self, other_scaler):
        self.min_val = np.minimum(self.min_val,other_scaler.min_val)
        self.max_val = np.maximum(self.max_val,other_scaler.max_val)
    
        
    def transform(self, data):
        return (data - self.min_val) * 1.0 / (self.max_val - self.min_val)
    
    def inverse_transform(self, data):
        if type(data)==np.ndarray:
            max_val = self.max_val.to_numpy().reshape(1,-1)
            min_val = self.min_val.to_numpy().reshape(1,-1)
            return data * (max_val - min_val) + min_val
        else:
            return data * (self.max_val - self.min_val) + self.min_val
    
def create_xyscaler(dfin,dfout):
    # xscaler=MinMaxScaler()
    xscaler=myscaler()
    _ = xscaler.fit_transform(dfin,name='inputs')
    #
    yscaler=myscaler()
    _ = yscaler.fit_transform(dfout,name='outputs')
    return xscaler, yscaler

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def dropout(x, p=0.03):
    return x * (np.random.binomial([np.ones(x.shape)],
                                  1-p)[0]) # * (1.0/(1-p)) # re-normalize the vector to compensate for 0's


def apply_augmentation(x, apply_aug=False,noise_sigma=0.03,dropout_ratio=0):
    np.random.seed(0)
    if apply_aug:
        x = jitter(x,sigma=noise_sigma)
        x = dropout(x,p=dropout_ratio)
    return x

def create_training_sets(dfin, dfout, calib_slice=slice('1940','2015'), valid_slice=slice('1923','1939'),
                         train_frac=None,
                         ndays=8,window_size=11,nwindows=10,
                         noise_sigma=0.,dropout_ratio=0.,
                         lead_time=0,lead_freq='D',
                         xscaler=None, yscaler=None,
                         keep_months_only=None):
    '''
    dfin is a dataframe that has sample (rows/timesteps) x nfeatures 
    dfout is a dataframe that has sample (rows/timesteps) x 1 label
    Both these data frames are assumed to be indexed by time with daily timestep

    This calls create_antecedent_inputs to create the CALSIM 3 way of creating antecedent information for each of the features

    Returns a tuple of two pairs (tuples) of calibration and validation training set where each set consists of input and output
    it also returns the xscaler and yscaler in addition to the two tuples above
    '''
    # create antecedent inputs aligned with outputs for each pair of dfin and dfout
    dfina,dfouta=[],[]
    # scale across all inputs and outputs
    if (xscaler is None) or (yscaler is None):
        xscaler,yscaler=create_xyscaler(dfin,dfout)
    if keep_months_only is not None:
        print('Training on %s seasons (%s) only' % (keep_months_only,'Jul. to Dec.' if keep_months_only == 'dry' else 'Jan. to Jun.'))
    
    for dfi,dfo in zip(dfin,dfout):
        dfi,dfo=synchronize(dfi,dfo,lead_time=lead_time,lead_freq=lead_freq)
        dfi,dfo=pd.DataFrame(xscaler.transform(dfi),dfi.index,columns=dfi.columns),pd.DataFrame(yscaler.transform(dfo),dfo.index,columns=dfo.columns)
        dfi,dfo=synchronize(create_antecedent_inputs(dfi,ndays=ndays,window_size=window_size,nwindows=nwindows),dfo)
        if keep_months_only == 'wet':
            dfi = dfi[(dfi.index.month>=7) & (dfi.index.month<=12)]
            dfo = dfo[(dfo.index.month>=7) & (dfo.index.month<=12)]
        elif keep_months_only == 'dry':
            dfi = dfi[(dfi.index.month>=1) & (dfi.index.month<=6)]
            dfo = dfo[(dfo.index.month>=1) & (dfo.index.month<=6)]
            
        dfina.append(dfi)
        dfouta.append(dfo)
        
    
    # split in calibration and validation slices
    if train_frac is None:
        dfins=[split(dfx,calib_slice,valid_slice) for dfx in dfina]
        dfouts=[split(dfy,calib_slice,valid_slice) for dfy in dfouta]
    else:
        train_sample_index = dfina[0].sample(frac=train_frac,random_state=0).index
        dfins=[(dfx.loc[dfx.index.isin(train_sample_index)],dfx.loc[~dfx.index.isin(train_sample_index)]) for dfx in dfina]
        dfouts=[(dfy.loc[dfy.index.isin(train_sample_index)],dfy.loc[~dfy.index.isin(train_sample_index)]) for dfy in dfouta]
        print('Randomly selecting %d samples for training, %d for test' % (dfins[0][0].shape[0],dfins[0][1].shape[0]))

    # append all calibration and validation slices across all input/output sets
    xallc,xallv=dfins[0]
    for xc,xv in dfins[1:]:
        xallc=np.append(apply_augmentation(xallc,noise_sigma=noise_sigma,dropout_ratio=dropout_ratio),xc,axis=0)
        xallv=np.append(xallv,xv,axis=0)
    yallc, yallv = dfouts[0]
    for yc,yv in dfouts[1:]:
        yallc=np.append(yallc,yc,axis=0)
        yallv=np.append(yallv,yv,axis=0)
    return (xallc,yallc),(xallv,yallv),xscaler,yscaler


def conv_filter_generator(ndays=7,window_size = 11, nwindows=10):
    w = np.zeros((1,ndays+nwindows*window_size,ndays+nwindows))
    for ii in range(ndays):
        w[0,ndays+nwindows*window_size-ii-1,ii] = 1
    for ii in range(nwindows):
        w[0,((nwindows-ii-1)*window_size):((nwindows-ii)*window_size),ndays+ii] = 1/window_size
    return w

def predict(model,dfx,xscaler,yscaler,columns=['prediction'],ndays=8,window_size=11,nwindows=10):
    dfx=pd.DataFrame(xscaler.transform(dfx),dfx.index,columns=dfx.columns)
    xx=create_antecedent_inputs(dfx,ndays=ndays,window_size=window_size,nwindows=nwindows)
    oindex=xx.index
    yyp=model.predict(xx)
    dfp=pd.DataFrame(yscaler.inverse_transform(yyp),index=oindex,columns=columns)
    return dfp

class ANNModel:
    '''
    model consists of the model file + the scaling of inputs and outputs
    '''
    def __init__(self,model,xscaler,yscaler):
        self.model=model
        self.xscaler=xscaler
        self.yscaler=yscaler
    def predict(self, dfin,columns=['prediction'],ndays=8,window_size=11,nwindows=10):
        return predict(self.model,dfin,self.xscaler,self.yscaler,columns=columns,ndays=ndays,window_size=window_size,nwindows=nwindows)

def save_model(location, model, xscaler, yscaler):
    '''
    save keras model and scaling to files
    '''
    joblib.dump((xscaler,yscaler),'%s_xyscaler.dump'%location)
    model.save('%s.h5'%location)

def load_model(location,custom_objects):
    '''
    load model (ANNModel) which consists of model (Keras) and scalers loaded from two files
    '''
    model=keras.models.load_model('%s.h5'%location,custom_objects=custom_objects)
    xscaler,yscaler=joblib.load('%s_xyscaler.dump'%location)
    return ANNModel(model,xscaler,yscaler)
