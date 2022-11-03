import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import time
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score,StratifiedKFold

lr = LinearRegression()
rgcv=RidgeCV()
eltcv=ElasticNetCV()
lasso=LassoCV()
rf =RandomForestRegressor()
gbdt=GradientBoostingRegressor()
xgb =XGBRegressor()
lgbm = LGBMRegressor()
models =[lr,rgcv,eltcv, lasso,rf,gbdt ,xgb,lgbm]
