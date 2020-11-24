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
forecast_out = 5
dataset['Prediction'] = dataset['Close'].shift(-forecast_out)
X = numpy.array(dataset.drop('Prediction', axis=1))[:-forecast_out]
y = numpy.array(dataset['Prediction'])[:-forecast_out]

# Fit the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False, stratify=None)
lr_model = LinearRegression(fit_intercept=True, n_jobs=-1)
lr_model.fit(X_train, y_train)
X_future = dataset.drop('Prediction', axis=1)[:-forecast_out]
X_future = X_future.tail(forecast_out)
X_future = numpy.array(X_future)
real_future_data = numpy.array(dataset.drop('Prediction', axis=1)[-forecast_out:])
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_poly_test = poly.fit_transform(X_future)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

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
plot_data = dataset[X.shape[0]:]
plot_data.rename({"Prediction": "Linear prediction"}, axis=1, inplace=True)
plot_data['Linear prediction'] = lr_prediction
plot_data['Poly prediction'] = poly_prediction
print(plot_data)
pyplot.figure(figsize=(8, 4))
pyplot.title('Model')
pyplot.xlabel('Days')
pyplot.ylabel('Close Price')
pyplot.plot(dataset['Close'])
pyplot.plot(plot_data[['Linear prediction', 'Poly prediction']])
pyplot.legend(['Real data', 'Linear prediction', 'Poly prediction'])
pyplot.show()
