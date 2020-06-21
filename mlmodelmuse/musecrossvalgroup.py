import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
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
from sklearn.model_selection import ShuffleSplit
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.preprocessing import label_binarize
cvknn=list()
cvlp=list()
cvdt=list()
cvlr=list()
cvsvm=list()
cvrf=list()
cvmlp=list()
cvgb=list()
cvab=list()
cvrbf=list()
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


df=pd.read_csv("/Users/mahima/research/allsubjects1000ms.csv")
df = df.dropna()
f=5
'''
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()
'''
def classification_report_with_accuracy_score(y_true, y_pred):

    print (classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

def evaluate_metric(model, x_cv, y_cv):
    return f1_score(y_cv, model.predict(x_cv), average='micro')




X=df.loc[1:5000,'1':'47']
X1=X
#X = np.array(X)
X=stats.zscore(X)

scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)

y=df.loc[1:5000,'48']
y1=y
X1=X.astype('float')
X1=X1.tolist()
#X=np.reshape(X,(46,9538))
#print(X.shape)
y1=y.astype('int')
y1=y1.tolist()
#y=np.reshape(y,(9538))
#print(y.shape)
groups=df.loc[1:5000,'group']
groups1=groups.astype('float')
groups1=groups1.tolist()

logo = LeaveOneGroupOut()


for train, test in logo.split(X1, y1, groups=groups1):
        X_train=X[train]
        X_test=X[test]
        y_train=y.iloc[train]
        y_test=y.iloc[test]
        '''
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf=LinearDiscriminantAnalysis(n_components=46)
        ida_em=clf.fit_transform(X,y)
        classes = ['red','blue','green']
        from matplotlib.colors import ListedColormap
        colours = ListedColormap(['r','b','g'])
        '''
        #scatter=plt.scatter(ida_em[1:300,0],ida_em[1:300,1],c=y[1:300].ravel(),alpha=0.6, cmap=colours)
        #plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        #plt.show()
        #X=ida_em


        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=6)
        neigh.fit(X_train,y_train)
        y_pred=neigh.predict(X_test)
        print('KNN Accuracy', accuracy_score(y_test, y_pred))
        print('KNN MCC',matthews_corrcoef(y_test, y_pred))
        print('KNN aucroc', roc_auc_score(y_test, neigh.predict_proba(X_test), multi_class='ovo'))

        cvknn.append(accuracy_score(y_test, y_pred))
        cvknn1.append(matthews_corrcoef(y_test, y_pred))
        cvknn2.append(roc_auc_score(y_test, neigh.predict_proba(X_test), multi_class='ovo'))


        #print(backward_elimination(X1,neigh))
        #resultsknn.append(accuracy_score(y_test, y_pred))




        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier()
        dt.fit(X_train,y_train)
        y_pred=dt.predict(X_test)
        print('DT MCC', matthews_corrcoef(y_test, y_pred))
        print('DT aucroc', roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovo'))
        print('Decision Tree Accuracy', accuracy_score(y_test, y_pred))
        cvdt.append(accuracy_score(y_test, y_pred))
        cvdt1.append(matthews_corrcoef(y_test, y_pred))
        cvdt2.append(roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovo'))



        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression(C=10, penalty='l2')
        logreg.fit(X_train,y_train)
        y_pred=logreg.predict(X_test)
        print('LR MCC', matthews_corrcoef(y_test, y_pred))
        print('LR aucroc', roc_auc_score(y_test, logreg.predict_proba(X_test), multi_class='ovo'))
        print('LR Accuracy', accuracy_score(y_test, y_pred))
        cvlr.append(accuracy_score(y_test, y_pred))
        cvlr1.append(matthews_corrcoef(y_test, y_pred))
        cvlr2.append(roc_auc_score(y_test, logreg.predict_proba(X_test), multi_class='ovo'))




        from sklearn.svm import SVC
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train,y_train)
        y_pred=svclassifier.predict(X_test)
        print('SVM MCC', matthews_corrcoef(y_test, y_pred))
        #print('SVM aucroc', roc_auc_score(y_test, svclassifier.predict_proba(X_test), multi_class='ovo'))
        print('SVM Accuracy', accuracy_score(y_test, y_pred))
        cvsvm.append(accuracy_score(y_test, y_pred))
        cvsvm1.append(matthews_corrcoef(y_test, y_pred))
        #cvsvm2.append(roc_auc_score(y_test, svclassifier.predict_proba(X_test), multi_class='ovo'))


        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(300,100), max_iter=1000,alpha=0.0001, learning_rate_init=0.01,activation='logistic')
        mlp.fit(X_train,y_train)
        y_pred=mlp.predict(X_test)
        print('NN MCC', matthews_corrcoef(y_test, y_pred))
        print('NN aucroc', roc_auc_score(y_test, mlp.predict_proba(X_test), multi_class='ovo'))
        print('NN Accuracy', accuracy_score(y_test, y_pred))
        cvmlp.append(accuracy_score(y_test, y_pred))
        cvmlp1.append(matthews_corrcoef(y_test, y_pred))
        cvmlp2.append(roc_auc_score(y_test, mlp.predict_proba(X_test), multi_class='ovo'))


        from sklearn.ensemble import RandomForestClassifier
        # Create a Gaussian Classifier

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print('RF MCC', matthews_corrcoef(y_test, y_pred))
        print('RF aucroc', roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))
        print('RF Accuracy', accuracy_score(y_test, y_pred))
        cvrf.append(accuracy_score(y_test, y_pred))
        cvrf1.append(matthews_corrcoef(y_test, y_pred))
        cvrf2.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))




 
        svclassifier = SVC(kernel='rbf', random_state=0, gamma=2, C=0.0001)
        # Train the classifier
        svclassifier.fit(X_train,y_train)
        y_pred=svclassifier.predict(X_test)
        print('SVM MCC', matthews_corrcoef(y_test, y_pred))
        #print('SVM aucroc', roc_auc_score(y_test, svclassifier.predict_proba(X_test), multi_class='ovo'))
        print('SVM RBF Accuracy', accuracy_score(y_test, y_pred))
        cvrbf.append(accuracy_score(y_test, y_pred))
        cvrbf1.append(matthews_corrcoef(y_test, y_pred))
        #cvrbf2.append(roc_auc_score(y_test, svclassifier.predict_proba(X_test), multi_class='ovo'))




        # Step 6: Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 10%
        clf = GradientBoostingClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print('GBf MCC', matthews_corrcoef(y_test, y_pred))
        print('GBC aucroc', roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))
        print('GBC Accuracy', accuracy_score(y_test, y_pred))
        cvgb.append(accuracy_score(y_test, y_pred))
        cvgb1.append(matthews_corrcoef(y_test, y_pred))
        cvgb2.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))





        # Step 5: Fit a AdaBoost model, " compared to "Decision Tree model, accuracy go up by 10%
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print('ABC Accuracy', accuracy_score(y_test, y_pred))
        cvgb.append(accuracy_score(y_test, y_pred))

def summarize_results(scores):
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
itr=[1,2]


print("KNN")
summarize_results(cvknn)
summarize_results(cvknn1)
summarize_results(cvknn2)

print("LR")
summarize_results(cvlr)
summarize_results(cvlr1)
summarize_results(cvlr2)




print("SVM")
summarize_results(cvsvm)
summarize_results(cvsvm1)
#summarize_results(cvsvm2)


print("NN")
summarize_results(cvmlp)
summarize_results(cvmlp1)
summarize_results(cvmlp2)


print("RF")
summarize_results(cvrf)
summarize_results(cvrf1)
summarize_results(cvrf2)


print("GBC")
summarize_results(cvgb)
summarize_results(cvgb1)
summarize_results(cvgb2)



print("SVM RBF")
summarize_results(cvrbf)










