import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier





col_names = ['top-left-square', 'top-middle-square', 'top-right-square', 
'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square','bottom-middle-square','bottom-right-square', 'Class']

import os
for dirname, _, filenames in os.walk('~/iia/projet'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('~/iia/projet/tic-tac-toe.data')
df
print(df.isnull().sum())

for i in df.columns:
  print(df[i].value_counts())
  print()

print(df.columns)
print (df.head())
x_num = pd.get_dummies(df[['x', 'x.1', 'x.2', 'x.3', 'o', 'o.1', 'x.4','o.2','o.3']]
,drop_first = True)

df.replace('negative',0,inplace=True)
df.replace('positive',1,inplace=True)

print(x_num.columns)
y = df.iloc[:,9].values

x_train, x_test, y_train, y_test = train_test_split(x_num,y,test_size = 0.33, random_state = 42)

st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.fit_transform(x_test)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =  42)
classifier.fit(x_train,y_train)

y_pred= classifier.predict(x_test)
print(y_pred)

cm = confusion_matrix(y_test,y_pred)
print(cm)

incorrect_pred = (y_test != y_pred).sum()
print(incorrect_pred)

metrics.accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
y_pred = linear_reg.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Erreur quadratique moyenne:", mse)

ridge_classifier = RidgeClassifier(alpha=1.0)
ridge_classifier.fit(x_train, y_train)

y_pred = ridge_classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)


plt.figure(figsize=(50,30))
tree.plot_tree(classifier,filled=True)
plt.show()
