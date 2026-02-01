import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from load_data import load_train_data
from preprocess import run_preprocessing
from models import train_linear_model, train_RF, train_lasso, train_elasticnet, train_ridge


def train_simple_models():

       # split data into training and testing set
       df_train = load_train_data()
       X, y = run_preprocessing(df_train)
       x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=5)

       # training basic linear model
       linear_model = train_linear_model(x_train, y_train)
       predict_lm = linear_model.predict(x_test)
       rmse_linear_model = np.sqrt(mean_squared_error(y_test, predict_lm))

       # checking basic linear stats
       print('Linear Regressor Train score: ', linear_model.score(x_train, y_train))
       print('Linear Regressor Test score: ', linear_model.score(x_test, y_test))
       print('Linear Regressor RMSE: ', rmse_linear_model)

       # training random forrest
       forest = train_RF(x_train, y_train)
       predict_forest = forest.predict(x_test)
       rmse_forest = np.sqrt(mean_squared_error(y_test, predict_forest))

       # checkiking random forest stats
       print('Forest Train score: ', forest.score(x_train, y_train))
       print('Forest Test score: ', forest.score(x_test, y_test))
       print('Forest RMSE: ', rmse_forest)

       # trying a mix of the two models
       mix_y = 0.5*predict_forest + 0.5*predict_lm
       rmse_ensemble = np.sqrt(mean_squared_error(y_test, mix_y))
       print('Mix RMSE: ', rmse_ensemble)


def train_further_models():

       # split data into training and testing set
       df_train = load_train_data()
       X, y = run_preprocessing(df_train)
       x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=5)

       # train ridge model
       ridge = train_ridge(x_train, y_train)
       print('Ridge train score: ', ridge.score(x_train, y_train))
       print('Ridge test score: ', ridge.score(x_test, y_test))
       print('Ridge RMSE: ', np.sqrt(mean_squared_error(y_test, ridge.predict(x_test))))

       # train lasso model
       lasso = train_lasso(x_train, y_train)
       print('Lasso train score: ', lasso.score(x_train, y_train))
       print('Lasso test score: ', lasso.score(x_test, y_test))
       print('Lasso RMSE: ', np.sqrt(mean_squared_error(y_test, lasso.predict(x_test))))

       # train elasticnet model
       elastic_net = train_elasticnet(x_train, y_train)
       print('ElasticNet train score: ', elastic_net.score(x_train, y_train))
       print('ElasticNet test score: ', elastic_net.score(x_test, y_test))
       print('ElasticNet RMSE: ', np.sqrt(mean_squared_error(y_test, elastic_net.predict(x_test))))

if __name__ == '__main__':
       train_simple_models()
       train_further_models()
