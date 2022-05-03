from scipy import stats
from PreProcessing import *

#################################################################
# ------------------ Base Methods ------------------ #
################################################################

col = 'Sales'

method = ['Average', 'Naive', 'Drift', 'SES']


def base_methods(train, test):
    for i in range(len(method)):
        if method[i] == 'Average':
            h_step_forecast, residual_err, forecast_err = avg_forecast_method(train, test)
        elif method[i] == 'Naive':
            h_step_forecast, residual_err, forecast_err = naive_forecast_method(train, test)
        elif method[i] == 'Drift':
            h_step_forecast, residual_err, forecast_err = drift_forecast_method(train, test)
        else:
            h_step_forecast, residual_err, forecast_err = ses_forecast_method(train, test, train[0], 0.5)

        squared_train_err = [number ** 2 for number in residual_err]
        squared_test_err = [number ** 2 for number in forecast_err]

        lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
            , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
            , "{0:.2f}".format(np.var(residual_err))
            , "{0:.2f}".format(np.var(forecast_err))
            , "{0:.2f}".format(q_value(residual_err, 15))
            , "{0:.2f}".format(np.var(residual_err) / np.var(forecast_err))
               ]

        # final_df = pd.DataFrame()

        if i == 0:
            final_df = pd.DataFrame(lst, columns=['Average'],
                                    index=['MSE_Fcast', 'MSE_Residual', 'Var_Pred', 'Var_Fcast', 'QValue_Residual'
                                        , 'Variance_Ratio'])
        else:
            final_df[method[i]] = lst

        y_axis, x_axis = auto_corr_func_lags(forecast_err, 20)
        span = 1.96 / np.sqrt(len(forecast_err))
        plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
        plt.stem(x_axis, y_axis)
        plt.legend()
        plt.xlabel("Lags")
        plt.ylabel("ACF")
        plt.title(f"{method[i]} Forecast ACF Plot")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(train, label='Training dataset')
        plt.plot([None for i in train] + [x for x in test], label='Testing dataset')
        plt.plot([None for i in train] + [x for x in h_step_forecast], label=f'{method[i]} method h-step prediction')
        plt.legend()
        plt.xlabel('Sample set')
        plt.ylabel('Sales')
        plt.title(f"{method[i]} Method Forecasting")
        plt.grid()
        plt.show()

    return final_df


final_df = base_methods(X_train[col], X_test[col])

#################################################################
# ------------------ Holt-Winters Method ------------------ #
################################################################

holtt = ets.ExponentialSmoothing(X_train[col], trend='add', damped_trend=False, seasonal='add',
                                 seasonal_periods=7).fit()

# holtt = ets.ExponentialSmoothing(X_train['Sales'], trend='add', damped_trend=False, seasonal=None).fit(optimized=True)
# holtt = ets.Holt(X_train['Sales'], damped_trend=False).fit(optimized=True)

holtf = holtt.forecast(steps=len(X_test[col]))

pred_train_holtf = holtt.predict(start=0, end=(len(X_train[col]) - 1))

holtf = pd.DataFrame(holtf, columns=['Forecast']).set_index(X_test.index)
pred_train_holtf = pd.DataFrame(pred_train_holtf, columns=['Residual']).set_index(X_train.index)

forecast_err_hw = []
residual_err_hw = []

for i in range(np.min(X_test.index), np.max(X_test.index)):
    forecast_err_hw.append(X_test[col][i] - holtf['Forecast'][i])

for j in range(len(X_train)):
    residual_err_hw.append(X_train[col][j] - pred_train_holtf['Residual'][j])

squared_train_err = [number ** 2 for number in residual_err_hw]
squared_test_err = [number ** 2 for number in forecast_err_hw]

lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
    , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
    , "{0:.2f}".format(np.var(residual_err_hw))
    , "{0:.2f}".format(np.var(forecast_err_hw))
    , "{0:.2f}".format(q_value(residual_err_hw, 15))
    , "{0:.2f}".format(np.var(residual_err_hw) / np.var(forecast_err_hw))]

final_df['Holt-Winter'] = lst

y_axis, x_axis = auto_corr_func_lags(forecast_err_hw, 20)
span = 1.96 / np.sqrt(len(forecast_err_hw))
plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
plt.stem(x_axis, y_axis)
plt.legend()
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.title("Holt-Winter Forecast ACF Plot")
plt.grid()
plt.show()

plt.figure()
plt.plot(X_train['Order Date'], X_train[col], label='Training dataset')
plt.plot(X_test['Order Date'], X_test[col], label='Testing dataset')
plt.plot(X_test['Order Date'], holtf, label='Holt-Winter h-step prediction')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title("Holt-Winter Method Forecasting")
# plt.xticks(X_train['Month'][::20])
plt.grid()
plt.show()


#######################################################################
# ------------------ Multiple Linear Regression ------------------ #
######################################################################

# Label encoding on categorical variables

def mapping(xx):
    dict = {}
    count = -1
    for x in xx:
        dict[x] = count + 1
        count = count + 1
    return dict


for i in ['City', 'State', 'Sub-Category', 'Ship Mode', 'Region', 'Segment', 'Category']:
    unique_tag = df_2[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    df_2[i] = df_2[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

df_2['norm_Sales'] = np.log(df_2['Sales'])

X = df_2[['Quantity', 'Discount', 'Profit']]

Y = df_2[['norm_Sales']]

#######################################################################
# ------------------ SVD and Condition Number ------------------ #
######################################################################

s, d, v = np.linalg.svd(X, full_matrices=True)

print(f"Singular value of dataframe are {d}")
print(f"Condition number for dataframe is {LA.cond(X)}")

# X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.20)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#######################################################################
# ------------------ Multiple Linear Regression ------------------ #
######################################################################

#########################################################################################
# ------------------ Feature selection - Backward stepwise regression ------------------ #
########################################################################################

X_train.drop(['Discount'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# X_train.drop(['Quantity'], axis=1, inplace=True)
# model = sm.OLS(y_train, X_train).fit()
# print(model.summary())

print("t-test p-values for all features: \n", model.pvalues)

print("#" * 100)

print("F-test for final model: \n", model.f_pvalue)

# stats.probplot(model.resid, dist="norm", plot= plt)
# plt.title("OLS Model Residuals Q-Q Plot")

col = ['Discount']

for i in col:
    X_test.drop(i, axis=1, inplace=True)

prediction = model.predict(X_train)

prediction = pd.DataFrame(prediction, columns=['Residual']).set_index(X_train.index)

forecast = model.predict(X_test)

forecast = pd.DataFrame(forecast, columns=['forecast']).set_index(X_test.index)

plt.figure()
plt.plot(y_train, label='Training Data')
plt.plot(y_test, label="Testing Data")
plt.plot(forecast['forecast'], label="MLR h-step Prediction")
plt.xlabel("Sample Space")
plt.ylabel("Sales")
plt.legend()
plt.title("Sales Dataset Predictions - Multiple Linear Regression")
plt.grid()
plt.tight_layout()
plt.show()

pred_error = np.subtract(y_train, np.asarray(prediction))

pred_error = pred_error.reset_index()
pred_error.drop('index', axis=1, inplace=True)

y_axis, x_axis = auto_corr_func_lags(pred_error['norm_Sales'], 20)
span = 1.96 / np.sqrt(len(pred_error['norm_Sales']))
plt.axhspan(-1 * span, span, alpha=0.2, color='blue')
plt.stem(x_axis, y_axis)
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.title("Residual Error ACF Plot - Multiple Linear Regression")
plt.grid()
plt.show()

forecast_error = np.subtract(y_test, np.asarray(forecast))

forecast_error = forecast_error.reset_index()
forecast_error.drop('index', axis=1, inplace=True)

Q = sm.stats.acorr_ljungbox(pred_error['norm_Sales'], lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]

lst = [np.round(np.sum(np.square(forecast_error['norm_Sales'])) / len(forecast_error), 2)
    , np.round(np.sum(np.square(pred_error['norm_Sales'])) / len(pred_error), 2)
    , np.round(np.var(pred_error['norm_Sales']), 2)
    , np.round(np.var(forecast_error['norm_Sales']), 2)
    , np.round(Q, 2)
    , np.round(np.var(pred_error['norm_Sales']) / np.var(forecast_error['norm_Sales']), 2)]

final_df['OLS_Model'] = lst

print(final_df.to_string())


#
# K = len(X_train.columns)  # number of columns for prediction
#
# T = len(y_train)
#
# var_pred = np.sqrt(np.sum(np.square(pred_error)) / (T - K - 1))
#
# print("Variance of residual error: ", var_pred[0])
#
# # Variance of forecast error
#
# T = len(y_test)
#
# var_fore = np.sqrt(np.sum(np.square(forecast_error)) / (T - K - 1))
#
# print("Variance of forecast error: ", var_fore[0])
