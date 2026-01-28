import numpy as np
import pandas as pd


def drop_outliers(df: pd.DataFrame):
    outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
    return df.drop(index=outliers.index)


def get_features(X):
    relevant_features = ['YearBuilt', 'TotRmsAbvGrd', 'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual', 'Neighborhood', 'KitchenQual', 'MSZoning']
    X = X[relevant_features]

    numeric_features = X.select_dtypes(include=['int', 'float']).columns.to_list()
    categorical_features = X.select_dtypes(include=['object']).columns.to_list()

    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna('missing')

    return X, categorical_features

def transform_y(y):
    return np.log1p(y)

def run_preprocessing(df: pd.DataFrame, target='SalePrice'):
    df = drop_outliers(df)
    X = df.drop(target, axis=1)
    y = df[target]
    X, categorical_features = get_features(X)
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    y = transform_y(y)
    return X, y