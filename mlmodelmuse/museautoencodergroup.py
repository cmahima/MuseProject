import keras
from keras import backend as K
import pandas as pd
#from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
from numpy import mean
import numpy as np
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

resultsknn=list()
resultsmlp=list()
resultsdt=list()
resultslr=list()
resultssvm=list()
resultsrbf=list()
resultsrf=list()
resultsgb=list()
cvknn1=list()
cvlp1=list()
cvdt1=list()
cvlr1=list()
cvsvm1=list()
cvrf1=list()
cvmlp1=list()
cvgb1=list()
cvab1=list()
cvrbf1=list()
cvknn2=list()
cvlp2=list()
cvdt2=list()
cvlr2=list()
cvsvm2=list()
cvrf2=list()
cvmlp2=list()
cvgb2=list()
cvab2=list()
cvrbf2=list()



df=pd.read_csv("/Users/mahima/research/allsubjects50ms.csv")
df = df.dropna()
X=df.loc[:,'1':'47']
#X = np.array(X)
#X=stats.zscore(X)

scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)

y=df.loc[:,'48']
X=X.astype('float')
y=y.astype('int')


x_train, x_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.1, random_state=42)

'''
x = Input(shape=[46])
y = Dense(10, activation='relu', name='encoder')(x)
y = Dense(46, name='decoder',activation='sigmoid')(y)

ae = Model(inputs=x, outputs=y)
ae.compile(optimizer='sgd', loss='mse')
ae.fit(x_train, x_train,epochs=300, batch_size=10, shuffle=True )


y = ae.get_layer('encoder').output
y = Dense(1, activation='sigmoid', name='predictions')(y)

classifier = Model(inputs=ae.inputs, outputs=y)
classifier.compile(loss='binary_crossentropy',optimizer='sgd')
classifier.fit(x_train, y_train)


'''
input_data = Input(shape=(46,))
#encoded1 = Dense(42, activation = 'relu')(input_data)
#encoded2 = Dense(36, activation = 'relu')(encoded1)
encoded3 = Dense(32, activation = 'relu')(input_data)
#encoded4 = Dense(24, activation = 'relu')(encoded3)
encoded5 = Dense(18, activation = 'relu')(encoded3)
#encoded6 = Dense(16, activation = 'relu')(encoded5)
#encoded7 = Dense(12, activation = 'relu')(encoded6)
#encoded8 = Dense(10, activation = 'relu')(encoded5)
encoded8 = Dense(5, activation = 'relu')(encoded5)


# Decoder Layers
#decoded1 = Dense(12, activation = 'relu')(encoded8)
#decoded2 = Dense(16, activation = 'relu')(decoded1)
decoded3 = Dense(18, activation = 'relu')(encoded8)
#decoded4 = Dense(24, activation = 'relu')(encoded5)
#decoded5 = Dense(24, activation = 'relu')(decoded3)
#decoded6 = Dense(32, activation = 'relu')(decoded3)
#decoded7 = Dense(36, activation = 'relu')(decoded6)
decoded8 = Dense(42, activation = 'relu')(decoded3)
decoded9 = Dense(46, activation = 'relu')(decoded8)

# this model maps an input to its reconstruction
autoencoder = Model(inputs=input_data, outputs=decoded9)

autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')
print(autoencoder.summary())

# training
autoencoder.fit(x_train, x_train,
            epochs=1000,
            batch_size=10,
            shuffle=True,
            validation_data=(x_test, x_test))  # need more tuning

encoder = Model(inputs = input_data, outputs = encoded8)
encoded_input = Input(shape = (5, ))
encoded_train = pd.DataFrame(encoder.predict(X))
encoded_train = encoded_train.add_prefix('feature_')

encoded_test = pd.DataFrame(encoder.predict(x_test))
encoded_test = encoded_test.add_prefix('feature_')
#encoded_train['target'] = y_train
encoded_test['target'] = y_test
y1=y

# test the autoencoder by encoding and decoding the test dataset
reconstructions_train = autoencoder.predict(X)
X1=reconstructions_train
groups=df.loc[:,'group']
groups1=groups.astype('float')
groups1=groups1.tolist()

logo = LeaveOneGroupOut()
#reconstructions_test = autoencoder.predict(x_test)
for train, test in logo.split(X1, y1, groups=groups1):
    X_train = X1[train]
    X_test = X1[test]
    y_train = y1.iloc[train]
    y_test = y1.iloc[test]


    from sklearn.ensemble import RandomForestClassifier



    #X_test=reconstructions_test


    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Random forest Accuracy', accuracy_score(y_test, y_pred))
    resultsrf.append(accuracy_score(y_test, y_pred))
    cvrf1.append(matthews_corrcoef(y_test, y_pred))
    cvrf2.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("KNN Accuracy", accuracy_score(y_test, y_pred))
    resultsknn.append(accuracy_score(y_test, y_pred))
    cvknn1.append(matthews_corrcoef(y_test, y_pred))
    cvknn2.append(roc_auc_score(y_test, neigh.predict_proba(X_test), multi_class='ovo'))



    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Decision Tree Accuracy", accuracy_score(y_test, y_pred))
    resultsdt.append(accuracy_score(y_test, y_pred))
    cvdt1.append(matthews_corrcoef(y_test, y_pred))
    cvdt2.append(roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovo'))


    from sklearn.linear_model import LogisticRegression
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    # fit the model with data
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("LR Accuracy", accuracy_score(y_test, y_pred))
    resultslr.append(accuracy_score(y_test, y_pred))
    cvlr1.append(matthews_corrcoef(y_test, y_pred))
    cvlr2.append(roc_auc_score(y_test, logreg.predict_proba(X_test), multi_class='ovo'))


    from sklearn.svm import SVC

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print('SVM Accuracy', accuracy_score(y_test, y_pred))
    resultssvm.append(accuracy_score(y_test, y_pred))
    cvsvm1.append(matthews_corrcoef(y_test, y_pred))


    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(60), max_iter=500)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print('NN Accuracy', accuracy_score(y_test, y_pred))
    resultsmlp.append(accuracy_score(y_test, y_pred))
    cvmlp1.append(matthews_corrcoef(y_test, y_pred))
    cvmlp2.append(roc_auc_score(y_test, mlp.predict_proba(X_test), multi_class='ovo'))

    svclassifier = SVC(kernel='rbf', random_state=0, gamma=6, C=1)
    # Train the classifier
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    from sklearn.metrics import accuracy_score

    print('SVM RBF Accuracy ', accuracy_score(y_test, y_pred))
    resultsrbf.append(accuracy_score(y_test, y_pred))


    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("GB Accuracy", accuracy_score(y_test, y_pred))
    resultsgb.append(accuracy_score(y_test, y_pred))
    cvgb1.append(matthews_corrcoef(y_test, y_pred))
    cvgb2.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))


def summarize_results(scores):
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))



print('KNN')
summarize_results(resultsknn)
summarize_results(cvknn1)
summarize_results(cvknn2)


print("LR")
summarize_results(resultslr)
summarize_results(cvlr1)
summarize_results(cvlr2)


print("SVM")
summarize_results(resultssvm)
summarize_results(cvsvm1)


print("NN")
summarize_results(resultsmlp)
summarize_results(cvmlp1)
summarize_results(cvmlp2)





print("RF")
summarize_results(resultsrf)
summarize_results(cvrf1)
summarize_results(cvrf2)


print("SVM rbf")
summarize_results(resultsrbf)




print("GBC")
summarize_results(resultsgb)
summarize_results(cvgb1)
summarize_results(cvgb2)





