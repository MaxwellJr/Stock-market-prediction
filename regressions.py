import pandas
import numpy
from matplotlib import pyplot
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read dataset and create dataframe
dataset = pandas.read_csv('NFLX_5Y.csv')
dataset = dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
dataset = dataset[['Close']]
forecast_out = 3
for i in range(forecast_out):
    dataset.insert(0, f'Close N-{i + 1}', dataset['Close'].shift(i + 1))
    dataset[f'Prediction N+{i + 1}'] = dataset['Close'].shift(-i - 1)
dataset = dataset[forecast_out:-forecast_out]
dataset.reset_index(drop=True, inplace=True)
print(dataset)
X = numpy.array(dataset.drop([f'Prediction N+{i + 1}' for i in range(forecast_out)], axis=1))
y = numpy.array(dataset[[f'Prediction N+{i + 1}' for i in range(forecast_out)]])

# Fit the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False, stratify=None)
lr_model = LinearRegression(fit_intercept=True, n_jobs=-1)
lr_model.fit(X_train, y_train)
X_future = numpy.array(X[-1:])
print('X_future')
print(X_future)
real_future_data = numpy.array(y[-1:])
print('Real future data')
print(real_future_data)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_future)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# Prediction
lr_prediction = lr_model.predict(X_future)
print("Linear model prediction:")
print(lr_prediction[-forecast_out:])
poly_prediction = poly_model.predict(X_poly_test)
print("Polynomial model prediction:")
print(poly_prediction[-forecast_out:])

# Calculating accuracy
print('[Linear] Mean Absolute Error:', mean_absolute_error(lr_prediction, real_future_data))  
print('[Linear] Mean Squared Error:', mean_squared_error(lr_prediction, real_future_data, squared=True))  
print('[Linear] Root Mean Squared Error:', mean_squared_error(lr_prediction, real_future_data, squared=False))
print('[Polynomial] Mean Absolute Error:', mean_absolute_error(poly_prediction, real_future_data))  
print('[Polynomial] Mean Squared Error:', mean_squared_error(poly_prediction, real_future_data, squared=True))  
print('[Polynomial] Root Mean Squared Error:', mean_squared_error(poly_prediction, real_future_data, squared=False))

# Plot
plot_data = pandas.DataFrame({'Prediction' : numpy.insert(real_future_data[0], 0, X_future[0][-1])})
plot_data.index += X.shape[0] - 1
plot_data.insert(1, 'Linear prediction', numpy.insert(lr_prediction[0], 0, X_future[0][-1]))
plot_data.insert(2, 'Poly prediction', numpy.insert(poly_prediction[0], 0, X_future[0][-1]))
print(plot_data)
pyplot.figure(figsize=(8, 4))
pyplot.title('Model')
pyplot.xlabel('Days')
pyplot.ylabel('Close Price')
pyplot.plot(dataset['Close'])
pyplot.plot(plot_data[['Prediction', 'Linear prediction', 'Poly prediction']], marker=".")
pyplot.legend(['Historical data', 'Real Values', 'Linear prediction', 'Poly prediction'])
pyplot.show()
