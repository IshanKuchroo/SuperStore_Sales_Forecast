import matplotlib.pyplot as plt

from PreProcessing import *

X_train, X_test = train_test_split(df_Sales, test_size=0.2, random_state=42, shuffle=False)

################################################################
# -------- Generalized Partial AutoCorrelation (GPAC) --------#
###############################################################

y_axis, x_axis = auto_corr_func_lags(df_Sales_2['Sales'], 30)

# gpac = GPAC(y_axis, 10, 10)

sns.heatmap(GPAC(y_axis, 10, 10), annot=True)
plt.title("Generalized Partial Auto-Correlation Function")
plt.show()

################################################################
# -------- Assigning Orders for models --------#
###############################################################

# non-seasonal orders
n_a = 1
n_b = 1
d = 0

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

##############################################################################
# -------- Auto-Regressive Moving Average Model (ARMA) --------#
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

forecast_error = X_test['Sales'] - x_test_hat

squared_train_err = [number ** 2 for number in residual_error]
squared_test_err = [number ** 2 for number in forecast_error]

lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
    , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
    , "{0:.2f}".format(np.var(residual_error))
    , "{0:.2f}".format(np.var(forecast_error))
    , "{0:.2f}".format(Q)
    , "{0:.2f}".format(np.var(residual_error) / np.var(forecast_error))]

armais_df = pd.DataFrame(lst, columns=[f'ARIMA({n_a},{d},{n_b})'],
                         index=['MSE_Fcast', 'MSE_Residual', 'Var_Pred', 'Var_Fcast', 'QValue_Residual',
                                'Variance_Ratio'])

# armais_df[f'ARIMA({n_a},{d},{n_b})'] = lst

model.plot_diagnostics(figsize=(16, 8))
plt.suptitle(f'ARIMA({n_a},{d},{n_b}) Diagnostic Analysis')
plt.show()

print("\n")

print(armais_df)

print("\n")
#######################################
# -------- ARMA Coefficients --------#
######################################

for i in range(1, n_a+1):
    print("The AR coefficients a{}".format(i), " is: ", -model.params[i])

for i in range(1, n_b+1):
    print("The MA coefficients b{}".format(i), " is: ", model.params[i + n_a])

print("\n")
################################################
# -------- ARMA Confidence Interval --------#
###############################################

for i in range(1, n_a+1):
    print("The confidence interval for a{}".format(i), " is: ", -model.conf_int()[i][0], " and ",
          -model.conf_int()[i][1])

for i in range(1, n_b+1):
    print("The confidence interval for b{}".format(i), " is: ", model.conf_int()[i + n_a][0], " and ",
          model.conf_int()[i + n_a][1])

print("\n")
################################################
# -------- Zero/Pole Cancellation --------#
###############################################

ar_params = np.array(-0.9375858036532047)
ma_params = np.array(-0.8780858153878965)
ar = np.r_[1, -ar_params]
ma = np.r_[1, ma_params]
ar_roots = np.roots(ar)
ma_roots = np.roots(ma)

print("Roots of AR process: ", ar_roots)
print("Roots of MA process: ", ma_roots)

plt.figure()
plt.plot(X_train['Sales'], label='Training dataset')
plt.plot(x_hat, label=f'ARMA(1,1) method 1-step prediction')
plt.legend()
plt.xlabel('Sample set')
plt.ylabel('Sales')
plt.title(f"ARMA(1,1) Forecasting")
plt.grid()
plt.show()