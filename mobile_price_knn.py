import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import config

train=pd.read_csv(config.train_csv_path)
test=pd.read_csv(config.test_csv_path)

test.drop('id',axis=1,inplace=True)

# 移除price_range欄位
X = train.drop('price_range',axis=1)
Y = train['price_range']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

# find the best K
accuracy = []
for k in range(1, round(math.sqrt(len(X_train)))):
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    accuracy.append(accuracy_score(Y_test, y_pred)) 

# 各k值的accuracy圖
k_range = range(1,round(math.sqrt(len(X_train))))
plt.plot(k_range, accuracy)
plt.show()

best_k = np.argmax(accuracy)
print('best k: ' + str(best_k))

# 用training data訓練knn模型
knn = KNeighborsClassifier(n_neighbors=best_k-1)
knn.fit(X_train,Y_train)
score = knn.score(X_test,Y_test)
print('knn score: ' + str(score))

# predict with X_test
pred1 = knn.predict(X_test)

# confusion_matrix
print()
print('test data confusion matrix:')
print(confusion_matrix(pred1,Y_test))

# Accuracy Precision Recall F1-score Support
print()
print('test data Performance metrics:')
print(classification_report(pred1,Y_test))

# predict with X_train
pred2 = knn.predict(X_train)

# confusion_matrix
print()
print('train data confusion matrix:')
print(confusion_matrix(pred2,Y_train))

# Accuracy Precision Recall F1-score Support
print()
print('train data Performance metrics:')
print(classification_report(pred2,Y_train))