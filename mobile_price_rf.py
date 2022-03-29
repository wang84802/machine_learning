import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import config

train=pd.read_csv(config.train_csv_path)
test=pd.read_csv(config.test_csv_path)

test.drop('id',axis=1,inplace=True)

# 移除price_range欄位
X = train.drop('price_range',axis=1)
Y = train['price_range']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

# 用training data訓練decisiontree模型
rfc = RandomForestClassifier(n_estimators=1000, n_jobs= -1, random_state=1, min_samples_leaf = 3) #1000棵decisiontrees
rfc.fit(X_train , Y_train)
score = rfc.score(X_test,Y_test)
print('rf score: ' + str(score))

# predict with X_test
pred1 = rfc.predict(X_test)

# confusion_matrix
print()
print('test data confusion matrix:')
print(confusion_matrix(pred1,Y_test))

# Accuracy Precision Recall F1-score Support
print()
print('test data Performance metrics:')
print(classification_report(pred1,Y_test))

# predict with X_train
pred2 = rfc.predict(X_train)

# confusion_matrix
print()
print('train data confusion matrix:')
print(confusion_matrix(pred2,Y_train))

# Accuracy Precision Recall F1-score Support
print()
print('train data Performance metrics:')
print(classification_report(pred2,Y_train))

# 列出各feature importances
imp=rfc.feature_importances_
names=list(X_train)
zip(imp,names)
imp, names= zip(*sorted(zip(imp,names)))
plt.barh(range(len(names)),imp,align='center')
plt.yticks(range(len(names)),names)
plt.xlabel('Importance of Features')
plt.ylabel('Features')
plt.title('Importance of Each Feature')
plt.show()