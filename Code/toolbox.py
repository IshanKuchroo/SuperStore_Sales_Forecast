##############################################
# -------- Import Libraries --------#
#############################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import scipy.signal as signal

print("#" * 100)
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

##############################################
# -------- Datetime Transformer --------#
#############################################

def datetime_transformer(df, datetime_vars):
    """
    The datetime transformer

    Parameters
    ----------
    df : the dataframe
    datetime_vars : the datetime variables

    Returns
    ----------
    The dataframe where datetime_vars are transformed into the following 6 datetime types:
    year, month, day, hour, minute and second
    """

    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year': lambda x: x.dt.year,
             'month': lambda x: x.dt.month,
             'day': lambda x: x.dt.day
             # ,
             # 'hour': lambda x: x.dt.hour,
             # 'minute': lambda x: x.dt.minute,
             # 'second': lambda x: x.dt.second
             }

    # Make a copy of df
    df_datetime = df.copy(deep=True)

    # For each variable in datetime_vars
    for var in datetime_vars:
        # Cast the variable to datetime
        df_datetime[var] = pd.to_datetime(df_datetime[var])

        # For each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # Add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' + datetime_type] = datetime_type_operator(df_datetime[var])

    # Remove datetime_vars from df_datetime
    # df_datetime = df_datetime.drop(columns=datetime_vars)

    return df_datetime


##############################################
# -------- Auto Correlation Function --------#
#############################################

def auto_corr_func_lags(y_tt, lags):
    ry = []
    l = []
    den = 0
    y_bar = np.mean(y_tt)

    for i in range(len(y_tt)):
        den += (y_tt[i] - y_bar) ** 2

    for k in range(0, lags + 1):
        num = 0
        for j in range(k, len(y_tt)):
            num += (y_tt[j] - y_bar) * (y_tt[j - k] - y_bar)

        acf = num / den

        ry.append(acf)

        l.append(k)

    ryy = ry[::-1]
    ry_f = ryy[:-1] + ry

    ll = l[::-1]
    ll = [li * -1 for li in ll]
    l_f = ll[:-1] + l

    return ry_f, l_f


##############################################
# -------- Rolling Mean and Variance --------#
#############################################

def cal_rolling_mean_var(df, col, mean_or_var):
    lst = []
    lst1 = []

    for i in range(0, len(df)):
        mean = 0
        var = 0
        if i == 0:
            mean += df[col][i]
            var = 0
        else:
            for j in range(0, i + 1):
                mean += df[col][j]
            mean = mean / (i + 1)

            for k in range(0, i + 1):
                var += (df[col][k] - mean) ** 2
            var = var / i

        lst.append(mean)
        lst1.append(var)

    if mean_or_var == 'mean':
        return lst
    else:
        return lst1


##############################################
# -------- Q=Value --------#
#############################################

def q_value(y_tt, lags):
    ry = []
    den = 0
    y_bar = np.mean(y_tt)

    for i in range(len(y_tt)):
        den += (y_tt[i] - y_bar) ** 2

    for k in range(0, lags + 1):
        num = 0
        for j in range(k, len(y_tt)):
            num += (y_tt[j] - y_bar) * (y_tt[j - k] - y_bar)

        acf = num / den

        ry.append(acf)

    # print(ry)
    ry = [number ** 2 for number in ry[1:]]

    q_value = np.sum(ry) * len(y_tt)

    return q_value


##############################################
# -------- Average Forecast Method --------#
#############################################

def avg_forecast_method(tr, tt):
    pred = []
    train_err = []
    test_err = []
    pred_test = []
    for i in range(1, len(tr)):
        p = 0
        for j in range(0, i):
            p += (tr[j])
        p = p / i
        e = tr[i] - p

        pred.append(p)
        train_err.append(e)

    p = np.sum(tr) / len(tr)

    for k in range(np.min(tt.index), np.max(tt.index)):
        test_err.append(tt[k] - p)
        pred_test.append(p)

    return pred_test, train_err, test_err


##############################################
# -------- Naive Forecast Method --------#
#############################################

def naive_forecast_method(tr, tt):
    pred = []
    train_err = []
    test_err = []
    pred_test = []
    for i in range(1, len(tr)):
        pred.append(tr[i - 1])
        e = tr[i] - tr[i - 1]
        train_err.append(e)

    # print(pred)
    # print(train_err)

    for k in range(np.min(tt.index), np.max(tt.index)):
        pred_test.append(tr[len(tr) - 1])
        test_err.append(tt[k] - tr[len(tr) - 1])

    # print(pred_test)
    # print(test_err)
    return pred_test, train_err, test_err


##############################################
# -------- Drift Forecast Method --------#
#############################################


def drift_forecast_method(tr, tt):
    pred = []
    train_err = []
    test_err = []
    pred_test = []
    for i in range(2, len(tr)):
        p = tr[i - 1] + (tr[i - 1] - tr[0]) / (i - 1)
        e = tr[i] - p

        pred.append(p)
        train_err.append(e)

    # print(pred)
    # print(train_err)

    for k in range(np.min(tt.index), np.max(tt.index)):
        p = tr[len(tr) - 1] + (k + 1) * (tr[len(tr) - 1] - tr[0]) / (len(tr) - 1)
        e = tt[k] - p

        pred_test.append(p)
        test_err.append(e)

    # print(pred_test)
    # print(test_err)
    return pred_test, train_err, test_err


##############################################################
# -------- Simple exponential smoothing Method --------#
#############################################################

def ses_forecast_method(tr, tt, l0, a):
    alpha = a
    l0 = l0
    pred = []
    train_err = []
    test_err = []
    pred_test = []
    for i in range(1, len(tr)):
        p = 0
        e = 0
        if i == 1:
            p = (alpha * tr[i - 1]) + ((1 - alpha) * l0)
            e = tr[i] - p
        else:
            p = (alpha * tr[i - 1]) + ((1 - alpha) * pred[i - 2])
            e = tr[i] - p

        pred.append(p)
        train_err.append(e)

    # print(pred)
    # print(train_err)

    for k in range(np.min(tt.index), np.max(tt.index)):
        p = (alpha * tr[len(tr) - 1]) + ((1 - alpha) * pred[len(pred) - 1])
        e = tt[k] - p

        pred_test.append(p)
        test_err.append(e)

    # print(pred_test)
    # print(test_err)
    return pred_test, train_err, test_err


##############################################################
# -------- Generalized Partial AutoCorrelation (GPAC) --------#
#############################################################


def GPAC(acfar, a, b):
    if a + b > int(len(acfar) / 2):
        det_df = pd.DataFrame()
        print("j and k values are more than number of lags")
        return det_df
    else:
        acfar = list(acfar)
        for k in range(1, a):
            det_lst = []
            for j in range(0, b):
                idx = acfar.index(1)
                if j > 0:
                    idx = idx + j
                lst = []
                num_lst = []

                if k == 1:
                    num_lst.append(acfar[idx + 1])
                else:
                    num_lst.append(acfar[idx + 1:(idx + 1) + k])

                for i in range(k):
                    lst.append(acfar[(idx + i) - (k - 1):(idx + i) + 1][::-1])

                den_mat = np.asarray(lst)
                den_det = np.linalg.det(den_mat)
                num_mat = den_mat
                num_mat[:, k - 1] = np.asarray(num_lst)
                num_det = np.linalg.det(num_mat)

                if np.abs(den_det) < 0.00001 or np.abs(num_det) < 0.00001:
                    num_det = 0.0

                det_lst.append(num_det / den_det)

            if k == 1:
                det_df = pd.DataFrame(det_lst, columns=[k])
            else:
                det_df[k] = det_lst

    return det_df


##############################################################
# -------- N-order Difference --------#
#############################################################

def difference(dataset, interval):
    diff = []

    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        if i == 1:
            diff.extend([0] * i)
        elif i == 2 and interval == 2:
            diff.extend([0] * i)
        elif i == 3 and interval == 3:
            diff.extend([0] * i)
        elif i == 4 and interval == 4:
            diff.extend([0] * i)
        elif i == 7 and interval == 7:
            diff.extend([0] * i)
        elif i == 12 and interval == 12:
            diff.extend([0] * i)

        diff.append(value)
    return diff


#################################################################
# ------------------ ADF Test for Stationary ------------------ #
################################################################


def ADF_Cal(x):
    result = adfuller(x, autolag='AIC')
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    if result[1] <= 0.05:
        print("Observation -> Sales is stationary")
    else:
        print("Observation -> Sales is non-stationary")
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


#################################################################
# ------------------ KPSS Test for Stationary ------------------ #
################################################################

def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='ct', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    if kpss_output[1] > 0.05:
        print("Observation -> Sales is stationary")
    else:
        print("Observation -> Sales is non-stationary")
    print(kpss_output)


################################################################################
# ------------------ Levenberg Marquardt algorithm ------------------ #
###############################################################################


def error(theta, n_a, y):
    den = [1.0] + theta[:n_a]
    num = [1.0] + theta[n_a:]

    if len(den) != len(num):
        if len(den) > len(num):
            for i in range(len(den) - len(num)):
                num += [0]
        else:
            for i in range(len(num) - len(den)):
                den += [0]

    sys = (den, num, 1)

    t, e = signal.dlsim(sys, y)

    return e


def LM_gradient(gtheta, n, n_a, y_train, e_prev):
    delta = 0.000001
    for i in range(n):
        gtheta[i] = gtheta[i] + delta

        e_out = error(gtheta, n_a, np.asarray(y_train))

        x = (e_prev - e_out) / delta

        # e_prev = e_out
        gtheta[i] = gtheta[i] - delta

        if i == 0:
            df = pd.DataFrame(x, index=range(0, len(x)))
        else:
            df[i] = x

    e_out = error(gtheta, n_a, np.asarray(y_train))

    sum_squared_error = np.matmul(e_out.T, e_out)

    X = df.to_numpy()

    A = np.matmul(X.T, X)

    g = np.matmul(X.T, e_out)

    return A, g, sum_squared_error


def LM_newton(A, g, n_theta, n, n_a, y_train, mu):
    I = np.identity(n)

    delta_theta = np.matmul(np.linalg.inv(A + mu * I), g)

    new_th = np.zeros(n).tolist()

    for j in range(len(n_theta)):
        new_th[j] = n_theta[j] + delta_theta[j][0]

    e_prev = error(new_th, n_a, np.asarray(y_train))

    sum_squared_error_new = np.matmul(e_prev.T, e_prev)

    return delta_theta, sum_squared_error_new, new_th


def LM_Algo(y_train, n_a, n_b):
    n = (n_a + n_b)  # total number of parameters

    theta = [0] * n

    N = len(y_train)
    e_prev = error(theta, n_a, np.asarray(y_train))

    A, g, sum_squared_error = LM_gradient(theta, n, n_a, y_train, e_prev)

    mu = 0.01
    mu_max = 10000000000

    delta_theta, sum_squared_error_new, theta_new = LM_newton(A, g, theta, n, n_a, y_train, mu)

    sse = [sum_squared_error]
    k_arr = [1, 2]
    sse.append(sum_squared_error_new)

    k = 1
    while k <= 100:
        if sum_squared_error_new < sum_squared_error:
            if np.linalg.norm(delta_theta) < 0.0001:  # 0.001
                theta_hat = theta_new
                var_hat = sum_squared_error_new / (N - n)
                cov_hat = var_hat[0][0] * np.linalg.inv(A)
                break
            else:
                theta = theta_new
                mu = mu / 10

        while sum_squared_error_new >= sum_squared_error:
            mu = mu * 10
            if mu > mu_max:
                print("Invalid Values")
                break

            delta_theta, sum_squared_error_new, theta_new = LM_newton(A, g, theta, n, n_a, y_train, mu)

        sse.append(sum_squared_error_new)
        k_arr.append(k + 2)

        k += 1

        theta = theta_new

        e_prev = error(theta, n_a, np.asarray(y_train))

        A, g, sum_squared_error = LM_gradient(theta, n, n_a, y_train, e_prev)
        delta_theta, sum_squared_error_new, theta_new = LM_newton(A, g, theta, n, n_a, y_train, mu)

    return theta_new, cov_hat, var_hat, sse, k_arr
