import keras
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from matplotlib import pyplot as plt


def pinball_loss_keras(q):
    def loss(y_true,y_pred):
        e=(y_true-y_pred)
        return K.mean(K.maximum(q*e,(q-1)*e))
    return loss

def smoothed_pinball_keras(q,eps):
    def loss(y_true,y_pred):
        e=y_true-y_pred
        esq=K.square(e)/(2*eps)
        elin=tf.subtract(K.abs(e),0.5*eps)
        eesq=esq*K.cast(K.less(K.abs(e),eps),'float32')
        eelin=elin*K.cast(K.greater_equal(K.abs(e),eps),'float32')
        ee=eelin+eesq
        return K.mean(tf.where(e>0,q*ee,(1-q)*ee))
    return loss

def simultaneous_loss_keras(quantiles,eps):
    def loss(y_true,y_pred):#rewrite loss hear as the mean of the means
        e=(y_true-y_pred)
        esq = K.square(e) / (2 * eps)
        elin = tf.subtract(K.abs(e), 0.5 * eps)
        eesq = esq * K.cast(K.less(K.abs(e), eps), 'float32')
        eelin = elin * K.cast(K.greater_equal(K.abs(e), eps), 'float32')
        ee = eelin + eesq
        e_upper=ee*quantiles
        e_lower=ee*(1-quantiles)
        q_e=tf.where(e>0,e_upper,e_lower)

        q_err=K.mean(q_e,axis=0)#Should remove the N dimension, reduce to a mean per quantile
        sum_q_err=K.mean(q_err)
        return sum_q_err

        #return K.mean(K.mean(K.maximum(quantiles*e,(quantiles-1)*e,axis=0),axis=0)) #

    return loss

def quantile_loss(q,y_true,y_pred):
    if len(q)==1:
        e=y_true-y_pred
        loss=np.mean(np.maximum(q*e,(q-1)*e))
    else:
        e = y_true - y_pred
        q_e=np.maximum(q*e,(q-1)*e,axis=1)
        q_err=np.mean(q_e,axis=0)
        sum_q_err=np.mean(q_err)
        loss=sum_q_err
    return loss

def q_loss(q,y_true,y_pred):
    e=y_true-y_pred
    loss=np.mean(np.maximum(q*e,(q-1)*e))
    return loss

def evaluate_model(df,quantiles,y_col,save_path=None,save=False,name=None):

    q_score = []
    for q in quantiles:
        q_score = np.append(q_score, q_loss(q, df[y_col], df[str(np.round(q, decimals=4))]))
    Y_test = df[y_col]
    # Percent Capture
    cap = np.sum(
        np.less_equal(df[y_col], df['0.9999']) * np.greater_equal(df[y_col], df['0.0001']) * 1) / len(
        df[y_col])
    pw=np.mean(df['0.9999']-df['0.0001'])
    for i in np.arange(0.1,1,0.1):
        upper=str(np.round(1-i/2,2))
        lower = str(np.round(i / 2, 2))
        cap=np.append(cap,np.sum(np.less_equal(df[y_col], df[upper]) * np.greater_equal(df[y_col], df[lower]) * 1) / len(
        df[y_col]))
        pw=np.append(pw,np.mean(df[upper]-df[lower]))

    ACE = np.mean(np.abs(cap-np.flip(np.arange(0.1, 1.05, 0.1))))
    capture_all_02 = np.sum(
        np.less_equal(df[y_col], df['0.9999']) * np.greater_equal(df[y_col], df['0.0001']) * np.less(df[y_col],
                                                                                                     0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)

    R = np.max(df[y_col]) - np.min(df[y_col])
    PINAW = np.array(pw) / (len(y_col) * R)

    average_PINAW = np.mean(PINAW)

    scores = {'Percent Capture': cap[0], 'Percent Capture (HH FRC < 0.2 mg/L)': capture_all_02,
              'Average Coverage Error': ACE, "Average Prediction Interval Normalized Average Width": average_PINAW,
              'Average Quantile Error':np.mean(q_score)}
    if save==True:
        scores_df=pd.Series(data=scores.values(),index=scores.keys())
        scores_df.to_csv(save_path+"_calibration_scores.csv")
    return

def CI_fig(CI,CI_02):
    x=np.arange(0.1,1.05,0.1)
    fig=plt.figure()
    plt.plot(x,x,c='k')
    plt.scatter(x,CI,c='b',label='overall')
    plt.scatter(x,CI_02,c='orange',label='HH FRC <0.2 mg/L')
    plt.ylabel('Confidence Interval')
    plt.xlabel('Percent Capture')

    return fig




