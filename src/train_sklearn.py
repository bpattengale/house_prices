import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from preprocess import run_preprocessing
from load_data import load_train_data


def train_models():

       # split data into training and testing set
       df_train = load_train_data()
       X, y = run_preprocessing(df_train)
       x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=5)

       # training basic linear model
       linear_model = LinearRegression()
       linear_model.fit(x_train, y_train)

       predict_lm = linear_model.predict(x_test)
       rmse_linear_model = np.sqrt(mean_squared_error(y_test, predict_lm))

       # checking basic linear stats
       print('Linear Regressor Train score: ', linear_model.score(x_train, y_train))
       print('Linear Regressor Test score: ', linear_model.score(x_test, y_test))
       print('Linear Regressor RMSE: ', rmse_linear_model)

       # training random forrest
       forest = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, random_state=5)
       forest.fit(x_train, y_train)

       predict_forest = forest.predict(x_test)
       rmse_forest = np.sqrt(mean_squared_error(y_test, predict_forest))

       # checkiking random forest stats
       print('Forest Train score: ', forest.score(x_train, y_train))
       print('Forest Test score: ', forest.score(x_test, y_test))
       print('Forest RMSE: ', rmse_forest)

       # trying a mix of the two models
       ensemble_y = 0.5*predict_forest + 0.5*predict_lm
       rmse_ensemble = np.sqrt(mean_squared_error(y_test, ensemble_y))
       print('Ensemble RMSE: ', rmse_ensemble)

if __name__ == '__main__':
       train_models()
