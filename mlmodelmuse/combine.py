import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('/Users/mahima/research/realdatapowerAF_7.csv')
df2 = pd.read_csv('/Users/mahima/research/realdatapowerTP_10.csv')
df3 = pd.read_csv('/Users/mahima/research/realdatapowerAF_8.csv')
df4 = pd.read_csv('/Users/mahima/research/realdatapowerTP_9.csv')
[m1, n1] = df1.shape
[m2, n2] = df2.shape
[m3, n3] = df3.shape
[m4, n4] = df4.shape
mini = min(m1, m2, m3, m4)
index = []
for i in range(0, mini):
    index.append(i)
# ,'TP_10alpha','TP_10beta','TP_10lowbeta',
# 'AF_8alpha','AF_8beta','AF_8lowbeta'
columns = ['AF_7alpha', 'AF_7beta', 'AF_7lowbeta', 'TP_10alpha', 'TP_10beta', 'TP_10lowbeta', 'AF_8alpha', 'AF_8beta',
           'AF_8lowbeta', 'TP_9alpha', 'TP_9beta', 'TP_9lowbeta', 'color']
dn = pd.DataFrame(index=index, columns=columns)
print(df2['rownumber'])

for i in range(mini):
    # print("hhhi")

    if (df1['rownumber'][i] in (df3['rownumber'])):
        dn['AF_7alpha'][i] = df1['alpha'][i]
        dn['AF_7beta'][i] = df1['beta'][i]
        dn['AF_7lowbeta'][i] = df1['lowbeta'][i]

        dn['TP_10alpha'][i] = df2['alpha'][i]
        dn['TP_10beta'][i] = df2['beta'][i]
        dn['TP_10lowbeta'][i] = df2['lowbeta'][i]

        dn['AF_8alpha'][i] = df3['alpha'][i]
        dn['AF_8beta'][i] = df3['beta'][i]
        dn['AF_8lowbeta'][i] = df3['lowbeta'][i]
        dn['color'][i] = df1['color'][i]

        dn['TP_9alpha'][i] = df4['alpha'][i]
        dn['TP_9beta'][i] = df4['beta'][i]
        dn['TP_9lowbeta'][i] = df4['lowbeta'][i]
        dn['color'][i] = df1['color'][i]

dn = dn.dropna()
sns.pairplot(dn, hue="color")
plt.show()
print(dn)
dn=dn.astype(float)
X = dn.loc[:, 'AF_7alpha':'TP_10lowbeta']
y = dn['color']
from sklearn.model_selection import train_test_split

y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
from sklearn.metrics import accuracy_score

print("KNN Accuracy", accuracy_score(y_test, y_pred))
# print(dn)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import accuracy_score

print("Decision Tree Accuracy", accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score

print("LR Accuracy", accuracy_score(y_test, y_pred))

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import accuracy_score

print('SVM Accuracy', accuracy_score(y_test, y_pred))

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
from sklearn.metrics import accuracy_score

print('NN Accuracy', accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

print('Random forest Accuracy', accuracy_score(y_test, y_pred))

svclassifier = SVC(kernel='rbf', random_state=0, gamma=6, C=1)
# Train the classifier
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import accuracy_score

print('SVM RBF Accuracy ', accuracy_score(y_test, y_pred))

print(dn)
# X = pd.get_dummies(X)
# target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X, y)
# use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
f = feat_importances.nlargest(10).index.tolist()
feat = [None] * (len(f) + 1)
print(len(feat))
for i in range(len(f)):
    feat[i] = f[i]
print(feat)
feat[len(f)] = 'color'
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
dn.to_csv('/Users/mahima/research/musecombineddata.csv')
cols = ['TP_10alpha', 'AF_7lowbeta', 'TP_10beta', 'TP_10lowbeta']
dn = dn[feat]
print(dn.head())

X = dn.loc[:, 'TP_10alpha':'TP_10lowbeta']
y = dn['color']
from sklearn.model_selection import train_test_split

y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
from sklearn.metrics import accuracy_score

print("KNN Accuracy", accuracy_score(y_test, y_pred))
# print(dn)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import accuracy_score

print("Decision Tree Accuracy of top features", accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score

print("LR Accuracy of top features", accuracy_score(y_test, y_pred))

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import accuracy_score

print('SVM Accuracy of top features', accuracy_score(y_test, y_pred))

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
from sklearn.metrics import accuracy_score

print('NN Accuracy of top features', accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

print('Random forest Accuracy of top features', accuracy_score(y_test, y_pred))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings

# Import packages to do the classifying
import numpy as np
from sklearn.svm import SVC


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


# Create a SVC classifier using a linear kernel
svclassifier = SVC(kernel='rbf', random_state=0, gamma=6, C=1)
# Train the classifier
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import accuracy_score

print('SVM RBF Accuracy of top features', accuracy_score(y_test, y_pred))

# Visualize the decision boundaries
print( dn.groupby('color').describe().unstack(1))