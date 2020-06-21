import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
import numpy as np
from numpy import std
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef

resultsknn=list()
resultsmlp=list()
resultsdt=list()
resultslr=list()
resultssvm=list()
resultsrbf=list()
resultsrf=list()

cvknn=list()
cvlp=list()
cvdt=list()
cvlr=list()
cvsvm=list()
cvrf=list()
cvmlp=list()

df=pd.read_csv("/Users/mahima/research/musefeaturesfinalbhavani50ms.csv")
df = df.dropna()
X=df.loc[1:1200,'1':'47']
#X = np.array(X)


scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)
y=df.loc[1:1200,'48']
X=X.astype('float')
y=y.astype('int')


#%%



def cross_val(classifier,splits,X,y):
    scores1= []
    cv = KFold(n_splits=splits, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):

        Xs_train, Xs_test, ys_train, ys_test = X[train_index], X[test_index], y.iloc[train_index], y.iloc[test_index]
        classifier.fit(Xs_train, ys_train)
        scores1.append(classifier.score(Xs_test, ys_test))
    return(np.mean(scores1))




for i in range(10):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(X_train, y_train)
    y_pred=neigh.predict(X_test)
    print("KNN Accuracy",matthews_corrcoef(y_test, y_pred))
    resultsknn.append(matthews_corrcoef(y_test, y_pred))




    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Decision Tree Accuracy", matthews_corrcoef(y_test, y_pred))
    resultsdt.append(matthews_corrcoef(y_test, y_pred))




    from sklearn.linear_model import LogisticRegression
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    # fit the model with data
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("LR Accuracy", matthews_corrcoef(y_test, y_pred))
    resultslr.append(matthews_corrcoef(y_test, y_pred))



    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print('SVM Accuracy', matthews_corrcoef(y_test, y_pred))
    resultssvm.append(matthews_corrcoef(y_test, y_pred))


    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(60), max_iter=500)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print('NN Accuracy', matthews_corrcoef(y_test, y_pred))
    resultsmlp.append(matthews_corrcoef(y_test, y_pred))


    from sklearn.ensemble import RandomForestClassifier
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Random forest Accuracy', matthews_corrcoef(y_test, y_pred))
    resultsrf.append(matthews_corrcoef(y_test, y_pred))



    svclassifier = SVC(kernel='rbf', random_state=0, gamma=6, C=1)
    # Train the classifier
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    from sklearn.metrics import accuracy_score

    print('SVM RBF Accuracy ', matthews_corrcoef(y_test, y_pred))
    resultsrbf.append(matthews_corrcoef(y_test, y_pred))


def summarize_results(scores):
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
itr=[1,2]
#itr=[1]


print("KNN")
summarize_results(resultsknn)
#plt.plot( itr,resultsknn, c = 'y', marker = "P", markersize=10, label='kNN')
#plt.show()
print("Decision Tree")
summarize_results(resultsdt)
#plt.plot( itr,resultsdt, c='g', marker="v", markersize=10, label='Decision Tree')
#plt.show()



print("LR")
summarize_results(resultslr)

#plt.plot( itr,resultsdt, c='m', marker=">", markersize=10, label='LR')
#plt.show()


print("SVM")
summarize_results(resultssvm)

#plt.plot( itr,resultssvm,  c='k', marker="D", markersize=10, label='SVM')
#plt.show()

print("NN")
summarize_results(resultsmlp)

#plt.plot( itr,resultsmlp, c='c', marker=">", markersize=10, label='NN')
#plt.show()



print("RF")
summarize_results(resultsrf)


#plt.plot(itr,resultsmlp, c='b', marker=".", markersize=10, label='RF')
plt.show()


print("SVM rbf")
summarize_results(resultsrbf)





