import numpy as np
import pandas as pd


def drop_outliers(df: pd.DataFrame):
    # dropping outliers
    outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
    return df.drop(index=outliers.index)


def get_features(X):
    # choosing features, removing nulls, and adding new features
    relevant_features = ['YearBuilt', 'YrSold', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual', 'Neighborhood', 'KitchenQual', 'MSZoning', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    X = X[relevant_features]

    numeric_features = X.select_dtypes(include=['int', 'float']).columns.to_list()
    categorical_features = X.select_dtypes(include=['object']).columns.to_list()

    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna('missing')

    X['BldgAge'] = X['YrSold'] - X['YearBuilt']
    X['TotalBath'] = X['FullBath'] + 0.5*X['HalfBath'] + X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X['TotalSF'] = X['1stFlrSF'] + X['2ndFlrSF']
    X.drop(['YrSold', 'YearBuilt', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

    X = pd.get_dummies(X, columns=categorical_features).astype(int)

    return X

def transform_y(y):
    # transform target with log
    return np.log1p(y)

def run_preprocessing(df: pd.DataFrame, target='SalePrice'):
    # run all preprocessing steps
    df = drop_outliers(df)
    X = df.drop(target, axis=1)
    y = df[target]
    X = get_features(X)
    y = transform_y(y)
    return X, y