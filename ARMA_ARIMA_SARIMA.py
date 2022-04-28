import matplotlib.pyplot as plt

from PreProcessing import *

y_axis, x_axis = auto_corr_func_lags(df_Sales['Sales'], 30)

################################################################
# -------- Generalized Partial AutoCorrelation (GPAC) --------#
###############################################################
# gpac = GPAC(y_axis, 10, 10)

sns.heatmap(GPAC(y_axis, 10, 10), annot=True)
plt.title("Generalized Partial Auto-Correlation Function")
plt.show()

################################################################
# -------- Assigning Orders to models --------#
###############################################################

# non-seasonal orders
n_a = 1
n_b = 1
d = 1

# seasonal orders
N_a = 0
N_b = 0
D = 0
s = 0

################################################################
# -------- Find chi-2 value --------#
###############################################################

lags = 20
alpha = 0.01
DOF = lags - n_a - n_b

chi_critical = chi2.ppf(1 - alpha, DOF)

print("Chi-critical value is: ", chi_critical)

print("\n")

################################################################
# -------- Auto-Regressive Moving Average Model (ARMA) --------#
###############################################################

model = ARMA(X_train['Sales'], (n_a, n_b)).fit(trend='c', disp=0)
# model = ARMA(X_train['pass_diff_01'], (n_a, n_b)).fit()

print(model.summary())

# Generate predictions

x_hat = model.predict(start=0, end=len(X_train['pass_diff_01']) - 1)

# Calculate residual errors

residual_error = X_train['Sales'] - x_hat
# residual_error = X_train['pass_diff_01'] - x_hat

# Find Q-value

Q = sm.stats.acorr_ljungbox(residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for ARMA residuals: ", Q)

chi_critical = chi2.ppf(1 - alpha, DOF)

print("Chi-critical value is: ", chi_critical)

if Q < chi_critical:
    print("As Q-value is less than chi-2 critical, Residual is white")
else:
    print("As Q-value is greater than chi-2 critical,Residual is NOT white")

print("Estimated variance of ARMA residual error: ", np.var(residual_error))

y_axis, x_axis = auto_corr_func_lags(residual_error, 20)

span = 1.96 / np.sqrt(len(residual_error))
plt.figure(figsize=(12, 8))
plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
plt.stem(x_axis, y_axis)
plt.legend()
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.title(f"ARMA({n_a},{n_b}) Auto-Correlation Plot")
plt.grid()
plt.show()

x_test_hat = model.forecast(steps=len(X_test))

forecast_error = X_test['Sales'] - x_test_hat[0]

# forecast_error = X_test['pass_diff_01'] - x_test_hat[0]

squared_train_err = [number ** 2 for number in residual_error]
squared_test_err = [number ** 2 for number in forecast_error]

lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
    , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
    , "{0:.2f}".format(np.var(residual_error))
    , "{0:.2f}".format(np.var(forecast_error))
    , "{0:.2f}".format(Q)
    , "{0:.2f}".format(np.var(residual_error) / np.var(forecast_error))]

armais_df = pd.DataFrame(lst, columns=[f'ARMA({n_a},{n_b})'],
                         index=['MSE_Fcast', 'MSE_Residual', 'Var_Pred', 'Var_Fcast', 'QValue_Residual',
                                'Variance_Ratio'])

# model.plot_diagnostics(figsize=(16, 8))
# plt.show()

print("\n")

##############################################################################
# -------- Auto-Regressive Integrated Moving Average Model (ARIMA) --------#
#############################################################################

model = ARIMA(X_train['Sales'].values, order=(n_a, d, n_b)).fit()

print(model.summary())

# Generate predictions

x_hat = model.predict(start=0, end=len(X_train['Sales']) - 1)

# Calculate residual errors

residual_error = X_train['Sales'] - x_hat

# Find Q-value

Q = sm.stats.acorr_ljungbox(residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for ARIMA residuals: ", Q)

if Q < chi_critical:
    print("As Q-value is less than chi-2 critical, Residual is white")
else:
    print("As Q-value is greater than chi-2 critical,Residual is NOT white")

print("Estimated variance of ARIMA residual error: ", np.var(residual_error))

y_axis, x_axis = auto_corr_func_lags(residual_error, 20)

span = 1.96 / np.sqrt(len(residual_error))
plt.figure(figsize=(12, 8))
plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
plt.stem(x_axis, y_axis)
plt.legend()
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.title(f"ARIMA({n_a},{d},{n_b}) Auto-Correlation Plot")
plt.grid()
plt.show()

x_test_hat = model.forecast(steps=len(X_test))

forecast_error = X_test['Sales'] - x_test_hat[0]

squared_train_err = [number ** 2 for number in residual_error]
squared_test_err = [number ** 2 for number in forecast_error]

lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
    , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
    , "{0:.2f}".format(np.var(residual_error))
    , "{0:.2f}".format(np.var(forecast_error))
    , "{0:.2f}".format(Q)
    , "{0:.2f}".format(np.var(residual_error) / np.var(forecast_error))]

armais_df[f'ARIMA({n_a},{d},{n_b})'] = lst

model.plot_diagnostics(figsize=(16, 8))
plt.suptitle(f'ARIMA({n_a},{d},{n_b})')
plt.show()

print("\n")

#######################################################################################
# -------- Seasonal Auto-Regressive Integrated Moving Average Model (SARIMA) --------#
######################################################################################

if N_a == 0 and N_b == 0 and D == 0:
    model = sm.tsa.statespace.SARIMAX(X_train['Sales'].values, order=(n_a, d, n_b)
                                      , enforce_stationarity=False
                                      , enforce_invertibility=False).fit()
else:
    model = sm.tsa.statespace.SARIMAX(X_train['Sales'].values, order=(n_a, d, n_b), seasonal_order=(N_a, D, N_b, s)
                                      , enforce_stationarity=False
                                      , enforce_invertibility=False).fit()

print(model.summary())

# Generate predictions

x_hat = model.predict(start=0, end=len(X_train['Sales']) - 1)

# Calculate residual errors

residual_error = X_train['Sales'] - x_hat

# Find Q-value

Q = sm.stats.acorr_ljungbox(residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for SARIMA residuals: ", Q)

if Q < chi_critical:
    print("As Q-value is less than chi-2 critical, Residual is white")
else:
    print("As Q-value is greater than chi-2 critical,Residual is NOT white")

print("Estimated variance of SARIMA residual error: ", np.var(residual_error))

y_axis, x_axis = auto_corr_func_lags(residual_error, 20)

span = 1.96 / np.sqrt(len(residual_error))
plt.figure(figsize=(12, 8))
plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
plt.stem(x_axis, y_axis)
plt.legend()
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.title(f"SARIMA({n_a},{d},{n_b})x({N_a}, {D}, {N_b}, {s}) Auto-Correlation Plot")
plt.grid()
plt.show()

x_test_hat = model.forecast(steps=len(X_test))

forecast_error = X_test['Sales'] - x_test_hat[0]

squared_train_err = [number ** 2 for number in residual_error]
squared_test_err = [number ** 2 for number in forecast_error]

lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
    , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
    , "{0:.2f}".format(np.var(residual_error))
    , "{0:.2f}".format(np.var(forecast_error))
    , "{0:.2f}".format(Q)
    , "{0:.2f}".format(np.var(residual_error) / np.var(forecast_error))]

if N_a == 0 and N_b == 0 and D == 0 and s == 0:
    armais_df[f"SARIMA({n_a},{d},{n_b})"] = lst
else:
    armais_df[f"SARIMA({n_a},{d},{n_b})x({N_a}, {D}, {N_b}, {s})"] = lst

print(armais_df)

model.plot_diagnostics(figsize=(16, 8))
if N_a == 0 and N_b == 0 and D == 0 and s == 0:
    plt.suptitle(f"SARIMA({n_a},{d},{n_b})")
else:
    plt.suptitle(f"SARIMA({n_a},{d},{n_b})x({N_a}, {D}, {N_b}, {s})")
plt.show()
