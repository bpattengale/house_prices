# House Prices ML

This is a machine learning project based on the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) competition. The goal is to predict house sale prices using the real world dataset provided. 

I am using this project to develop and practice my machine learning skills. My plan is to start off with simple models as a baseline and add more advanced models and neural networks later on to be able to compare the differences.


## Exploratory Data Analysis

An initial analysis of the data was done in [notebooks/data_exploration.ipynb](https://github.com/bpattengale/house_prices/blob/main/notebooks/data_exploration.ipynb) to understand:

- Target variable distribution and skewness
- Missing values
- Correlations of variables with the target 'SalePrice'
- Identify outliers
- Boxplots of categorical variables to compare the variation in 'SalePrice'


## Preprocessing and Initial Models

So far I have done a bit of feature engineering in the [src/preprocess.py](https://github.com/bpattengale/house_prices/blob/main/src/preprocess.py) to pick out some features to train the simple models. I will be adding more features later on for better training. 

The first models I have added are a simple linear regression model and a random forest regressor both from sklearn in [src/train_sklearn.py](https://github.com/bpattengale/house_prices/blob/main/src/train_sklearn.py) and also a weighted mix of the two. So far it looks like the random forest model is dramatically overfitting the data so I will have to do some tuning to improve the accuracy. New models I have added are the Ridge, Lasso, and ElasticNet also from sklearn. So far they appear to be describing the data well but will need tuning of parameters to maxamize accuracy.

## TODO

- add and adjust which features go into model training
- tune parameters of existing models
- experiment with new models and ensembles