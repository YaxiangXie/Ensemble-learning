import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
def visualize(score_list):
    train_score_list = [i['train_score'] for i in score_list]
    valid_score_list = [i['valid_score'] for i in score_list]
    plt.plot(train_score_list, label='train_score')
    plt.plot(valid_score_list, label='valid_score')
    plt.grid()
    plt.legend()

df = pd.read_csv('./BlackFriday.csv')
X = df.loc[:, df.columns != 'Purchase'].copy()
y = df.loc[:, df.columns == 'Purchase'].copy()

for c in ['Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']:
    X[c] = LabelEncoder().fit_transform(X[c])

X = SimpleImputer().fit_transform(X)
y = y.values.reshape(-1)

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1)

print(X.shape, y.shape)
print(train_X.shape, train_y.shape)
print(valid_X.shape, valid_y.shape)

class MyGradientBoostingRegressor:
    
    def __init__(self, n_estimators=100, lr=0.1, max_depth=3, verbose=False):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.verbose = verbose
        
        self.estimator_list = None
        self.is_first = True
        self.F = None
        self.score_list = list()
        
    def fit(self, train_X, train_y):
        self.estimator_list = list()
        self.F = np.zeros_like(train_y, dtype=float)
        
        for i in range(1, self.n_estimators + 1):
            # get negative gradients
            neg_grads = train_y - self.F
            base = DecisionTreeRegressor(max_depth=self.max_depth)
            base.fit(train_X, neg_grads)
            train_preds = base.predict(train_X)
            self.estimator_list.append(base)
            
            if self.is_first:
                self.F = train_preds
                self.is_first = False
            else:
                self.F += self.lr * train_preds
                
            train_score = r2_score(train_y, self.F)
            valid_preds = self.predict(valid_X)
            valid_score = r2_score(valid_y, valid_preds)
            iter_score = dict(iter=i, train_score=train_score, valid_score=valid_score)
            self.score_list.append(iter_score)
            if self.verbose:
                print(iter_score)
                
    def predict(self, X):
        F = np.zeros_like(len(X), dtype=float)
        is_first = True
        for base in self.estimator_list:
            preds = base.predict(X)
            if is_first:
                F = preds
                is_first = False
            else:
                F += self.lr * preds
        return F
    
model = GradientBoostingRegressor(n_estimators=300, max_depth=5)
model.fit(train_X, train_y)
print('GradientBoostingRegressor train_score: {:.4f} valid_score: {:.4f}'.format(
    r2_score(train_y, model.predict(train_X)), r2_score(valid_y, model.predict(valid_X))))

model = MyGradientBoostingRegressor(n_estimators=300, max_depth=5)
model.fit(train_X, train_y)
print('MyGradientBoostingRegressor train_score: {:.4f} valid_score: {:.4f}'.format(
    r2_score(train_y, model.predict(train_X)), r2_score(valid_y, model.predict(valid_X))))