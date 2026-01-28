import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from preprocess import run_preprocessing
from load_data import load_train_data

def train_model():

       df_train = load_train_data()

       X, y = run_preprocessing(df_train)

       x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

       model = LinearRegression()

       model.fit(x_train, y_train)

       y_predict = model.predict(x_test)
       rmse = np.sqrt(mean_squared_error(y_test, y_predict))

       print('Train score:')
       print(model.score(x_train, y_train))

       print('Test score:')
       print(model.score(x_test, y_test))

       print('RMSE:')
       print(rmse)


if __name__ == '__main__':
       train_model()