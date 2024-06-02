import numpy as np
from numpy import random
random.seed(256)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn import preprocessing  #標準化
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def adaBoostingTrain(X, y, n):
      D = np.ones(len(X))
      alphaArray = [] # 紀錄決策樹權重
      weakLearners = []
      for i in range(n):
            # 歸一化 weight
            D /= np.sum(D) 
            weakLearner = DecisionTreeClassifier(max_depth = 1)
            weakLearner.fit(X, y, sample_weight = D)
            weakLearners.append(weakLearner) #record

            y_pred = weakLearner.predict(X)
            # 計算誤差及加權誤差
            errorArray = [int(x) for x in (y_pred != y)]
            errorArray2 =[i if i>0 else -1 for i in errorArray] # change to 1/-1 set
            errorSum = np.sum(D * (y_pred != y))

            # 更新權重
            alpha = 0.5 * (np.log((1 - errorSum) / float(errorSum)))
            alphaArray.append((alpha)) #record
            #調整訓練資料權重
            D *= np.exp([float(x) * alpha for x in errorArray2])
      
      return weakLearners, alphaArray

def strongLearner_pred(X_test,weakLearners, alphaArray):
      test_pred = np.zeros(len(X_test))
      #整合學習機
      for weakLearner, alpha in zip(weakLearners, alphaArray):
            y_pred = weakLearner.predict(X_test)
            y_pred = [i if i>0 else -1 for i in y_pred]
            test_pred += np.multiply(alpha, y_pred)
            
      final_pred = np.sign(test_pred)
      return final_pred


if __name__ == '__main__':
      #Load Data set and standardize
      Dataset, target = make_moons(n_samples = 1000, noise = 0.05, random_state = 47)
      target = 2 * target - 1
      print(Dataset.shape) #(1000, 2)
      print(target.shape)  #(1000)

      X = preprocessing.scale(Dataset)

      #################################
      X_train, X_test, y_train, y_test = train_test_split(X, target , train_size = 700, stratify = target, random_state = 47)
      print(f"訓練資料樣本數: {len(X_train)}")  #700
      print(f"測試資料樣本數: {len(X_test)}")  #300

      '''
      題目條件:
      1、 20 顆決策樹
      2、 每顆樹深度為 1 
      '''
      weakLearners, alphaArray = adaBoostingTrain(X_train, y_train, 20)
      #各顆決策樹權重展示
      x = [i for i in range(1, 21)]
      plt.scatter(x, alphaArray)
      plt.xlim(0, 20)
      plt.ylim(0, 1)
      x_major_locator = 1
      plt.xlabel('estimator index')
      plt.ylabel('estimator weight')
      plt.show()
      ####################################
      #計算對測試資料之準確率
      y_test_pred = strongLearner_pred(X_test, weakLearners, alphaArray)
      score = accuracy_score(y_test, y_test_pred)

      print(f"Strong Learner accuracy: {score:.4f}")



    
    