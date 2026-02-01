from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

def train_linear_model(x_train, y_train):
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    return lm

def train_ridge(x_train, y_train, alpha=55, max_iter=10000, random_state=3):
    ridge = Ridge(random_state=random_state)
    ridge.fit(x_train, y_train)
    return ridge

def train_lasso(x_train, y_train, alpha=0.0005, max_iter=10000, random_state=4):
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
    lasso.fit(x_train, y_train)
    return lasso

def train_elasticnet(x_train, y_train, alpha=0.0015, l1_ratio=0.35, max_iter=10000, random_state=5):
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state)
    en.fit(x_train, y_train)
    return en

def train_RF(x_train, y_train, n_estimators=300, max_depth=20, min_samples_split=5, random_state=6):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    rf.fit(x_train, y_train)
    return rf