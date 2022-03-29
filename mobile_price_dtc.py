import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus
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
dtc = DecisionTreeClassifier()
dtc.fit(X_train , Y_train)
score = dtc.score(X_test,Y_test)
print('dtc score: ' + str(score))

# 匯出decisiontree pdf
# dot_data=tree.export_graphviz(dtc,out_file=None)
# graph=pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf('mobile_price.pdf')

# predict with X_test
pred1 = dtc.predict(X_test)

# confusion_matrix
print()
print('test data confusion matrix:')
print(confusion_matrix(pred1,Y_test))

# Accuracy Precision Recall F1-score Support
print()
print('test data Performance metrics:')
print(classification_report(pred1,Y_test))

# predict with train.csv
pred2 = dtc.predict(X_train)

# confusion_matrix
print()
print('train data confusion matrix:')
print(confusion_matrix(pred2,Y_train))

# Accuracy Precision Recall F1-score Support
print()
print('train data Performance metrics:')
print(classification_report(pred2,Y_train))