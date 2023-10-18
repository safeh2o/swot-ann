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

def evaluate_model(df, quantiles,x_col,y_col,save_path=None,save=False,name=None):
    q_score = []
    for q in quantiles:
        q_score = np.append(q_score, q_loss(q, df[y_col], df[str(np.round(q, decimals=4))]))
    Y_test=df[x_col]
    # Percent Capture
    capture_all = np.sum(
        np.less_equal(df[y_col], df['0.9999']) * np.greater_equal(df[y_col], df['0.0001']) * 1) / len(
        df[y_col])
    capture_90 = np.sum(np.less_equal(df[y_col], df['0.95']) * np.greater_equal(df[y_col], df['0.05']) * 1) / len(
        df[y_col])
    capture_80 = np.sum(np.less_equal(df[y_col], df['0.9']) * np.greater_equal(df[y_col], df['0.1']) * 1) / len(
        df[y_col])
    capture_70 = np.sum(np.less_equal(df[y_col], df['0.85']) * np.greater_equal(df[y_col], df['0.15']) * 1) / len(
        df[y_col])
    capture_60 = np.sum(np.less_equal(df[y_col], df['0.8']) * np.greater_equal(df[y_col], df['0.2']) * 1) / len(
        df[y_col])
    capture_50 = np.sum(np.less_equal(df[y_col], df['0.75']) * np.greater_equal(df[y_col], df['0.25']) * 1) / len(
        df[y_col])
    capture_40 = np.sum(np.less_equal(df[y_col], df['0.7']) * np.greater_equal(df[y_col], df['0.3']) * 1) / len(
        df[y_col])
    capture_30 = np.sum(np.less_equal(df[y_col], df['0.65']) * np.greater_equal(df[y_col], df['0.35']) * 1) / len(
        df[y_col])
    capture_20 = np.sum(np.less_equal(df[y_col], df['0.6']) * np.greater_equal(df[y_col], df['0.4']) * 1) / len(
        df[y_col])
    capture_10 = np.sum(np.less_equal(df[y_col], df['0.55']) * np.greater_equal(df[y_col], df['0.45']) * 1) / len(
        df[y_col])

    capture = np.array(
        [capture_10, capture_20, capture_30, capture_40, capture_50, capture_60, capture_70, capture_80, capture_90,
         capture_all])
    x = np.arange(0.1, 1.05, 0.1)
    CI_sumsquares = np.sum((capture - x) ** 2)

    capture_all_02 = np.sum(
        np.less_equal(df[y_col], df['0.9999']) * np.greater_equal(df[y_col], df['0.0001']) * np.less(df[y_col],
                                                                                                           0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_90_02 = np.sum(
        np.less_equal(df[y_col], df['0.95']) * np.greater_equal(df[y_col], df['0.05']) * np.less(df[y_col],
                                                                                                       0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_80_02 = np.sum(
        np.less_equal(df[y_col], df['0.9']) * np.greater_equal(df[y_col], df['0.1']) * np.less(df[y_col],
                                                                                                     0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_70_02 = np.sum(
        np.less_equal(df[y_col], df['0.85']) * np.greater_equal(df[y_col], df['0.15']) * np.less(df[y_col],
                                                                                                       0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_60_02 = np.sum(
        np.less_equal(df[y_col], df['0.8']) * np.greater_equal(df[y_col], df['0.2']) * np.less(df[y_col],
                                                                                                     0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_50_02 = np.sum(
        np.less_equal(df[y_col], df['0.75']) * np.greater_equal(df[y_col], df['0.25']) * np.less(df[y_col],
                                                                                                       0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_40_02 = np.sum(
        np.less_equal(df[y_col], df['0.7']) * np.greater_equal(df[y_col], df['0.3']) * np.less(df[y_col],
                                                                                                     0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_30_02 = np.sum(
        np.less_equal(df[y_col], df['0.65']) * np.greater_equal(df[y_col], df['0.35']) * np.less(df[y_col],
                                                                                                       0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_20_02 = np.sum(
        np.less_equal(df[y_col], df['0.6']) * np.greater_equal(df[y_col], df['0.4']) * np.less(df[y_col],
                                                                                                     0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)
    capture_10_02 = np.sum(
        np.less_equal(df[y_col], df['0.55']) * np.greater_equal(df[y_col], df['0.45']) * np.less(df[y_col],
                                                                                                       0.2) * 1) / np.sum(
        np.less(df[y_col], 0.2) * 1)


    capture_02 = np.array(
        [capture_10_02, capture_20_02, capture_30_02, capture_40_02, capture_50_02, capture_60_02, capture_70_02,
         capture_80_02, capture_90_02, capture_all_02])
    CI_sumsquares_02 = np.sum((capture_02 - x) ** 2)

    rank = []
    rank_02 = []
    preds_df = df.drop([y_col, x_col], axis=1)
    for a in range(0, len(Y_test)):
        observation = df[y_col].iloc[a]
        forecast = preds_df.iloc[a].to_numpy()
        n_lower = np.sum(np.greater(observation, forecast))
        n_equal = np.sum(np.equal(observation, forecast))
        deviate_rank = np.random.random_integers(0, n_equal)
        rank = np.append(rank, n_lower + deviate_rank)
        if observation < 0.2:
            rank_02 = np.append(rank_02, n_lower + deviate_rank)

    RH = np.histogram(rank, bins=101)[0]
    RH_02 = np.histogram(rank_02, bins=101)[0]
    delta = np.sum((RH - (len(df[y_col]) / (1 + len(preds_df.columns)))) ** 2)
    delta_0 = len(preds_df.columns) * len(df[y_col]) / (1 + len(preds_df.columns))
    delta_score = delta / delta_0

    delta_02 = np.sum((RH_02 - (np.sum(np.less(df[y_col], 0.2)) / (1 + len(preds_df.columns)))) ** 2)
    delta_0_02 = len(preds_df.columns) * len(df[y_col]) / (1 + len(preds_df.columns))
    delta_score_02 = delta_02 / delta_0_02

    alpha = np.zeros((len(df[y_col]), 1 + len(preds_df.columns)))
    beta = np.zeros((len(df[y_col]), 1 + len(preds_df.columns)))
    low_outlier = 0
    high_outlier = 0

    for a in range(0, len(Y_test)):
        observation = df[y_col].iloc[a]
        forecast = (np.sort(preds_df.iloc[a].to_numpy()))
        for b in range(1, len(preds_df.columns)):
            if observation > forecast[b]:
                alpha[a, b] = forecast[b] - forecast[b - 1]
                beta[a, b] = 0
            elif forecast[b] > observation > forecast[b - 1]:
                alpha[a, b] = observation - forecast[b - 1]
                beta[a, b] = forecast[b] - observation
            else:
                alpha[a, b] = 0
                beta[a, b] = forecast[b] - forecast[b - 1]
        # overwrite boundaries in case of outliers
        if observation < forecast[0]:
            beta[a, 0] = forecast[0] - observation
            low_outlier += 1
        if observation > forecast[1 - len(preds_df.columns)]:
            alpha[a, len(preds_df.columns)] = observation - forecast[1 - len(preds_df.columns)]
            high_outlier += 1

    alpha_bar = np.mean(alpha, axis=0)
    beta_bar = np.mean(beta, axis=0)
    g_bar = alpha_bar + beta_bar
    o_bar = beta_bar / (alpha_bar + beta_bar)

    if low_outlier > 0:
        o_bar[0] = low_outlier / len(df[y_col])
        g_bar[0] = beta_bar[0] / o_bar[0]
    else:
        o_bar[0] = 0
        g_bar[0] = 0
    if high_outlier > 0:
        o_bar[len(preds_df.columns)] = high_outlier / len(df[y_col])
        g_bar[len(preds_df.columns)] = alpha_bar[len(preds_df.columns)] / o_bar[len(preds_df.columns)]
    else:
        o_bar[len(preds_df.columns)] = 0
        g_bar[len(preds_df.columns)] = 0

    p_i = np.arange(0 / len(preds_df.columns), 1 + (1 / len(preds_df.columns))*0.5, (1 / len(preds_df.columns)))

    CRPS = np.sum(g_bar * ((1 - o_bar) * (p_i ** 2) + o_bar * ((1 - p_i) ** 2)))
    Reli = np.sum(g_bar * ((o_bar - p_i) ** 2))



    scores = {'Percent Capture': capture_all*100, 'Percent Capture (HH FRC < 0.2 mg/L)': capture_all_02*100,
              'PI Reliability Score': CI_sumsquares,
              'Delta': delta_score,  'CRPS': CRPS, 'Reli': Reli,'Average Quantile Score':np.mean(q_score)}
    if save==True:
        CI_x = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        scores_df=pd.Series(data=scores.values(),index=scores.keys())
        scores_df.to_csv(save_path+"_scores.csv")
        np.savetxt(save_path+ "_PI_reliability_scatter.csv",
                   np.transpose([CI_x, capture, capture_02]),
                   delimiter=',', header="Prediction Interval,Overall Capture,Capture of Values below 0.2", comments='')

    return scores_df

def CI_fig(CI,CI_02):
    x=np.arange(0.1,1.05,0.1)
    fig=plt.figure()
    plt.plot(x,x,c='k')
    plt.scatter(x,CI,c='b',label='overall')
    plt.scatter(x,CI_02,c='orange',label='HH FRC <0.2 mg/L')
    plt.ylabel('Confidence Interval')
    plt.xlabel('Percent Capture')

    return fig




