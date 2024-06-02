import numpy as np
from numpy import random
random.seed(46)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn import preprocessing  #標準化    
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize_scalar


class GradientBoostingModel:
    def __init__(self):
        self.Tree_num = 20
        self.max_depth = 1
        self.estimatorList = []
        self.TreeList = []
    #微分
    def grad(self, x, y):
        return -x / (1 + np.exp(np.multiply(x , y)))
    #logistic loss
    def logistic_loss(self, x, y):
        loss = np.log(1 + np.exp(np.multiply(-x, y)))
        return loss

    def gradientBoosting(self, X, y):
        F = np.zeros(len(X))
        for i in range(self.Tree_num):
            y_new = -self.grad(y, F)
            #以新target，訓練新的決策樹
            weakLearner_new = DecisionTreeRegressor(max_depth = self.max_depth)
            weakLearner_new.fit(X, y_new)
            self.TreeList.append(weakLearner_new) #record
            y_pred_next = weakLearner_new.predict(X)
            # 步幅使用線搜尋法，誤差度量使用logistic loss
            delta = lambda a: np.mean(self.logistic_loss(y, F + a * y_pred_next))
            step = minimize_scalar(delta, bounds = [0, 1], method = 'bounded')
            a = step.x
            self.estimatorList.append(a)

            F += a * y_pred_next
        print(self.estimatorList)
        return self.estimatorList , self.TreeList
    
    #最終學習機預測
    def gradientBoosting_pred(self, X_test, y_test):
        F = np.zeros(len(X_test))
        #整合學習機
        for weakLearner, a in zip(self.TreeList, self.estimatorList):
            y_pred_next = weakLearner.predict(X_test)
            F += np.multiply(a, y_pred_next)

        F = [1 if i > 0 else -1 for i in F]
        return F

if __name__ == '__main__':
    #Load Data set and standardize
    Dataset, target = make_moons(n_samples = 1000, noise = 0.05, random_state = 47)
    print(Dataset.shape) #(1000, 2)
    print(target.shape)  #(1000)

    X = preprocessing.scale(Dataset)
    target = 2 * target - 1 #change to 1/-1 set
    #################################
    X_train, X_test, y_train, y_test = train_test_split(X, target , train_size = 700, stratify = target, random_state = 47)
    print(f"訓練資料樣本數: {len(X_train)}")  #700
    print(f"測試資料樣本數: {len(X_test)}")  #300

    '''
    題目條件:
    1、 20 顆決策樹
    2、 每顆樹深度為 1 
    '''
    model = GradientBoostingModel()
    estimatorList, TreeList = model.gradientBoosting(X_train, y_train)
    #各顆決策樹權重展示
    x = [i for i in range(1, 21)]
    plt.scatter(x, estimatorList)
    plt.xlim(0, 20)
    plt.ylim(0, 1)
    x_major_locator = 1
    plt.xlabel('estimator index')
    plt.ylabel('estimator weight')
    plt.show()
    
    ####################################
    #計算對測試資料之準確率
    y_test_pred = model.gradientBoosting_pred(X_test, y_test)

    score = accuracy_score(y_test, y_test_pred)

    print(f"Strong Learner accuracy: {score:.4f}")



    
    