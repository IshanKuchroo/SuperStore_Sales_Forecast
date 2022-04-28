##############################################
# Import Libraries #
#############################################

# Standard
import pandas as pd
import numpy as np
import numpy.linalg as LA
import scipy.signal as signal
from scipy import stats
from datetime import datetime
from toolbox import *

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import rcParams

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from scipy.stats import chi2
import pmdarima as pm


#############################################################
# ------------------ Reading Dataset ------------------ #
############################################################

def data_processing():
    df = pd.read_csv("train.csv")

    print("Original Dataset: \n", df.head())
    print("Dataset Statistics: \n", df.describe())

    df_2 = df.copy()

    #####################################################################
    # ------------------ Drop ID columns ------------------ #
    ####################################################################

    df_2.drop(["Row ID", "Order ID", "Customer ID", "Product ID"], axis=1, inplace=True)

    #####################################################################
    # ------------------ Correcting the Date format ------------------ #
    ####################################################################

    df_2["Order Date"] = pd.to_datetime(df_2["Order Date"], format='%m/%d/%Y')
    df_2["Ship Date"] = pd.to_datetime(df_2["Ship Date"], format='%m/%d/%Y')

    df_2 = df_2.sort_values(by=['Order Date']).reset_index(drop=True)

    #############################################################
    # ------------------ Filling NaN values ------------------ #
    ############################################################

    print("Columns with NA values - Before removal:")
    for i in df_2.columns:
        if df_2[i].isna().sum() > 0:
            print(df_2[i].name, ":", df_2[i].isna().sum())
        else:
            continue
    print("\n")

    # df_2['Postal Code'] = df_2['Postal Code'].fillna(5401)

    print("Columns with NA values - After filling values:")
    for i in df_2.columns:
        if df_2[i].isna().sum() > 0:
            print(df_2[i].name, ":", df_2[i].isna().sum())
        else:
            continue
    print("\n")

    #################################################################################
    # ------------------ Creating Sales Uni-variate Dataframe ------------------ #
    ################################################################################

    df_Sales = df_2.groupby("Order Date").sum()[["Sales"]]

    # df_Sales = pd.DataFrame(df_Sales['Sales'].resample('D').mean())
    # df_Sales = df_Sales.interpolate(method='linear')

    df_Sales = df_Sales.reset_index()

    #############################################################
    # ------------------ Sales v/s Time ------------------ #
    ############################################################

    df_Sales["Month"] = pd.DatetimeIndex(df_Sales["Order Date"]).month
    df_Sales["Year"] = pd.DatetimeIndex(df_Sales["Order Date"]).year

    df_Sales["Year_Month"] = df_Sales["Year"].astype(str) + "-" + df_Sales["Month"].astype(str)
    df_Sales["Year_Month"] = pd.to_datetime(df_Sales["Year_Month"]).dt.date

    plt.figure(figsize=(12, 8))
    plt.plot(df_Sales['Order Date'], df_Sales['Sales'])
    plt.xlabel("Time")
    plt.ylabel("Sales(USD)")
    plt.legend()
    plt.title("Sales per Order Date")
    plt.grid()
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(12, 8))

    for i in range(4):
        ax1 = fig.add_subplot(2, 2, i + 1)
        if i == 0:
            ax1.plot(df_Sales['Order Date'], df_Sales['Sales'])
            ax1.set_title("Sales Data - By Date")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Sales(USD)")
            plt.grid()
        elif i == 1:
            ax1.plot(df_Sales.groupby("Year_Month")["Sales"].sum())
            ax1.set_title("Sales Data - By Year/Month")
            ax1.set_xlabel("Year_Month")
            ax1.set_ylabel("Sales(USD)")
            plt.grid()
        elif i == 2:
            ax1.plot(df_Sales.groupby("Year")["Sales"].sum())
            ax1.set_title("Sales Data - By Year")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Sales(USD)")
            plt.grid()
        elif i == 3:
            ax1.plot(df_Sales.groupby("Month")["Sales"].sum())
            ax1.set_title("Sales Data - By Month")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Sales(USD)")
            plt.grid()

    plt.suptitle("Superstore Sales Data v/s Time")
    plt.tight_layout()
    plt.show()

    df_Sales.drop("Month", axis=1, inplace=True)
    df_Sales.drop("Year", axis=1, inplace=True)
    df_Sales.drop("Year_Month", axis=1, inplace=True)

    ##############################################
    # ------------------ ACF/PACF ------------------ #
    #############################################

    y_axis, x_axis = auto_corr_func_lags(df_Sales['Sales'], 20)
    span = 1.96 / np.sqrt(len(df_Sales['Sales']))
    plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
    plt.stem(x_axis, y_axis)
    plt.legend()
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.title("Auto-Correlation Plot for Sales")
    plt.grid()
    plt.tight_layout()
    plt.show()

    def ACF_PACF_Plot(y, lags):
        acf = sm.tsa.stattools.acf(y, nlags=lags)
        pacf = sm.tsa.stattools.pacf(y, nlags=lags)
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.title('ACF/PACF of the raw data')
        plot_acf(y, ax=plt.gca(), lags=lags)
        plt.subplot(212)
        plot_pacf(y, ax=plt.gca(), lags=lags)
        fig.tight_layout(pad=3)
        plt.show()

    ACF_PACF_Plot(df_Sales['Sales'], 150)

    #############################################################
    # ------------------ Correlation Matrix ------------------ #
    ############################################################

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_2.corr(), annot=True)
    plt.tight_layout()
    plt.show()

    ####################################################################################
    # ------------------ Rolling Mean and Variance Stationary Test------------------ #
    ###################################################################################

    # df_Sales['Sales_Rolling_Mean'] = cal_rolling_mean_var(df_Sales, 'Sales', 'mean')
    # df_Sales['Sales_Rolling_Variance'] = cal_rolling_mean_var(df_Sales, 'Sales', 'var')
    #
    # plt.figure(figsize=(12, 8))
    # fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax[0].plot(df_Sales['Order Date'], df_Sales['Sales_Rolling_Mean'], 'b', label='Rolling Mean Sales')
    # ax[0].legend()
    # ax[0].set_title('Stationary test of Sales - Rolling Mean')
    # ax[0].set_xlabel('Order Date')
    # ax[0].set_ylabel('Sales')
    # ax[0].grid()
    # ax[1].plot(df_Sales['Order Date'], df_Sales['Sales_Rolling_Variance'], 'r', label='Rolling Variance Sales')
    # ax[1].legend()
    # ax[1].set_title('Stationary test of Sales - Rolling Variance')
    # ax[1].set_xlabel('Order Date')
    # ax[1].set_ylabel('Sales')
    # ax[1].grid()
    # plt.tight_layout()
    # plt.show()

    ###############################################################################
    # ------------------ ADF Test and KPSS Test for Stationary ------------------ #
    ##############################################################################

    print("ADF Stats for Sales: \n")
    ADF_Cal(df_Sales['Sales'])
    print("\n")

    print("KPSS Stats for Sales: \n")
    kpss_test(df_Sales['Sales'])
    print("\n")

    ###############################################################################################
    #   Rolling Mean and Variance, ADF and KPSS Test for Stationary  - Difference Data #
    ##############################################################################################

    df_Sales['pass_diff_01'] = difference(df_Sales['Sales'], 1)

    ACF_PACF_Plot(df_Sales['pass_diff_01'], 150)

    # plt.figure(figsize=(12, 8))
    # plt.hist(df_Sales['Sales'])
    # plt.xlabel("Sales(USD)")
    # plt.title("Distribution of Sales")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(12, 8))
    # plt.hist(df_Sales['pass_diff_01'])
    # plt.xlabel("Sales(USD)")
    # plt.title("Distribution of Sales - 1st order differencing")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # df_Sales['Pass1_Rolling_Mean'] = cal_rolling_mean_var(df_Sales, 'pass_diff_01', 'mean')
    # df_Sales['Pass1_Rolling_Variance'] = cal_rolling_mean_var(df_Sales, 'pass_diff_01', 'var')
    #
    # plt.figure(figsize=(50, 50))
    # fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax[0].plot(df_Sales['Order Date'], df_Sales['Pass1_Rolling_Mean'], 'b', label='Rolling Mean')
    # ax[0].legend()
    # ax[0].set_title('Rolling Mean - 1st order difference')
    # ax[0].set_xlabel('Order Date')
    # ax[0].set_ylabel('Sales')
    # ax[0].grid()
    # ax[1].plot(df_Sales['Order Date'], df_Sales['Pass1_Rolling_Variance'], 'r', label='Rolling Variance')
    # ax[1].legend()
    # ax[1].set_title('Rolling Variance - 1st order difference')
    # ax[1].set_xlabel('Order Date')
    # ax[1].set_ylabel('Sales')
    # ax[1].grid()
    # plt.tight_layout()
    # plt.show()

    print("ADF Stats for Sales - 1st order differencing: \n")
    ADF_Cal(df_Sales['pass_diff_01'])
    print("\n")

    print("KPSS Stats for Sales - 1st order differencing: \n")
    kpss_test(df_Sales['pass_diff_01'])
    print("\n")

    return df_2, df_Sales


df_2, df_Sales = data_processing()

#################################################################
# ------------------ Time Series Decomposition ------------------ #
################################################################


Temp = df_Sales['Sales']
Temp = pd.Series(np.array(Temp), index=pd.date_range('2015-01-03', periods=len(Temp), freq='D'))
STL = STL(Temp)
res = STL.fit()

plt.figure(figsize=(20, 10))
fig = res.plot()
plt.legend()
plt.grid()
plt.suptitle("Sales STL decomposition")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

print("#" * 100)

f_t = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(T) + np.array(R)))

print(f"The strength of trend for this data set is {f_t}")

f_s = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S) + np.array(R)))

print(f"The strength of seasonality for this data set is {f_s}")

print("\n")

#############################################################
# ------------------ Split Data into train and test ------------------ #
############################################################

X_train, X_test = train_test_split(df_Sales, test_size=0.2, random_state=42, shuffle=False)
