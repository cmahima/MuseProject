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
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.metrics import matthews_corrcoef

cvknn=list()
cvlp=list()
cvdt=list()
cvlr=list()
cvsvm=list()
cvrf=list()
cvmlp=list()
cvgb=list()
cvab=list()
cvxgb=list()
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

df=pd.read_csv("/Users/mahima/research/musefeaturesfinalchitraaallfeatures50ms.csv")
df = df.dropna()
f=10
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

def forward_feature_selection(x_train, x_cv, y_train, y_cv, n,model):
    feature_set = []
    for num_features in range(n):
        metric_list = [] # Choose appropriate metric based on business problem
        model = model # You can choose any model you like, this technique is model agnostic
        for feature in x_train.columns:
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                model.fit(x_train[f_set], y_train)
                metric_list.append((evaluate_metric(model, x_cv[f_set], y_cv), feature))

        metric_list.sort(key=lambda x : x[0], reverse = True) # In case metric follows "the more, the merrier"
        feature_set.append(metric_list[0][1])
    return feature_set

def backward_elimination(X,model):
    cols = list(X.columns)
    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X[cols]
        X_1 = add_constant(X_1)
        model = sm.model(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(selected_features_BE)

y=df.loc[1:1200,'87']
y1=y
#df1=stats.zscore(df.loc[:,'1':'86'])
X=df.loc[1:1200,'1':'86']
X1=X
X = np.array(X)
X=stats.zscore(X)
print(X)
scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)


X=X.astype('float')
y=y.astype('int')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf=LinearDiscriminantAnalysis(n_components=46)
ida_em=clf.fit_transform(X,y)
classes = ['red','blue','green']
from matplotlib.colors import ListedColormap
colours = ListedColormap(['r','b','g'])

#scatter=plt.scatter(ida_em[1:300,0],ida_em[1:300,1],c=y[1:300].ravel(),alpha=0.6, cmap=colours)
#plt.legend(handles=scatter.legend_elements()[0], labels=classes)
#plt.show()
#X=ida_em

'''
def cross_val(classifier,splits,X,y):
    scores1= []
    cv = KFold(n_splits=splits, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):

        Xs_train, Xs_test, ys_train, ys_test = X[train_index], X[test_index], y.iloc[train_index], y.iloc[test_index]
        classifier.fit(Xs_train, ys_train)
        scores1.append(classifier.score(Xs_test, ys_test))
    return(np.mean(scores1))

'''
folds=5

for i in range(1):
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier

    clf = RandomForestClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    features = forward_feature_selection(X_train, X_test, y_train, y_test, f, clf)
    X = X1[features]
    #y=y[11:1200]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(clf, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(clf, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(clf, X, y, cv=cv,scoring='f1_micro')
    print("RF recall",rec)
    print("RF Precision",pre)
    print("RF F1",f1)
    print('Random forest Accuracy', scores)
    aucrf=cross_val_score(clf, X, y, cv=cv,scoring='roc_auc_ovo')
    print("RF  AUC",aucrf)
    cvrf=scores
    #df1=df.loc[1:10,'1':'86']
    #df1 = df1.astype('float')
    #df1=df1[features]
    #df1=pd.read_csv("/Users/mahima/PycharmProjects/preprocessmuse/test.csv")
    #X2 = X1[1:10][features]
    #df1=df1[features]
    #print(df1)
    print(features)
    #result=clf.predict(df1)
    #print("prediction is", result)

    nested_score = cross_val_score(clf, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print('matthews_corrcoef',nested_score)
    cvrf1=aucrf
    cvrf2=nested_score

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=6)
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    features=forward_feature_selection(X_train, X_test, y_train, y_test,f, neigh)
    X=X1[features]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(neigh, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(neigh, X, y, cv=cv,scoring='recall_weighted')
    pre=cross_val_score(neigh, X, y, cv=cv,scoring='precision_weighted')
    f1=cross_val_score(neigh, X, y, cv=cv,scoring='f1_weighted')
    #mccknn=cross_val_score(neigh, X, y, cv=cv,scoring='matthews_corrcoef')
    aucknn=cross_val_score(neigh, X, y, cv=cv,scoring='roc_auc_ovo')
    print("KNN recall",rec)
    print("KNN Precision",pre)
    print("KNN F1",f1)
    print("KNN Accuracy",scores)
    print("AUC KNN",aucknn)
    cvknn=scores
    cvknn1=aucknn
    nested_score = cross_val_score(neigh, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print("matthews_corrcoef",nested_score)
    cvknn2=nested_score

    #print(backward_elimination(X1,neigh))
    #resultsknn.append(accuracy_score(y_test, y_pred))

    clf = GradientBoostingClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    features=forward_feature_selection(X_train, X_test, y_train, y_test,f, clf)
    X=X1[features]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(clf, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(clf, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(clf, X, y, cv=cv,scoring='f1_micro')
    print("GBC recall",rec)
    print("GBC Precision",pre)
    print("GBC F1",f1)
    print('Gradient BOOST Accuracy', scores)
    cvgb=scores
    aucgb=cross_val_score(clf, X, y, cv=cv,scoring='roc_auc_ovo')
    print("GB  AUC",aucgb)
    nested_score = cross_val_score(clf, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print('matthews_corrcoef',nested_score)
    cvgb1=aucgb
    cvgb2=nested_score



    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    features=forward_feature_selection(X_train, X_test, y_train, y_test,f, dt)
    X=X1[features]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(dt, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(dt, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(dt, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(dt, X, y, cv=cv,scoring='f1_micro')
    print("Decision Tree recall",rec)
    print("Decision Tree Precision",pre)
    print("Decision Tree F1",f1)
    print("Decision Tree Accuracy", scores)
    aucdt=cross_val_score(dt, X, y, cv=cv,scoring='roc_auc_ovo')
    print("Decision Tree AUC",aucdt)
    cvdt=scores
    nested_score = cross_val_score(dt, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print('matthews_corrcoef',nested_score)
    cvdt1=aucdt
    cvdt2=nested_score

    #resultsdt.append(accuracy_score(y_test, y_pred))




    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(C=10, penalty='l2')
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=42)
    features = forward_feature_selection(X_train, X_test, y_train, y_test, f, logreg)
    X = X1[features]
    #, solver='liblinear')
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(logreg, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(logreg, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(logreg, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(logreg, X, y, cv=cv,scoring='f1_micro')
    #mcclr=cross_val_score(logreg, X, y, cv=cv,scoring='matthews_corrcoef')
    #auclr=cross_val_score(logreg, X, y, cv=cv,scoring='cross_val_score')
    print("LR recall",rec)
    print("LR Precision",pre)
    print("LR F1",f1)
    print("LR Accuracy", scores)
    auclr=cross_val_score(logreg, X, y, cv=cv,scoring='roc_auc_ovo')
    print("LR  AUC",auclr)
    cvlr=scores
    nested_score = cross_val_score(logreg, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print('matthews_corrcoef',nested_score)
    #print(forward_feature_selection(X_train, X_test, y_train, y_test,10, logreg))
    #resultslr.append(accuracy_score(y_test, y_pred))
    cvlr1=auclr
    cvlr2=nested_score

    '''

    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    features=forward_feature_selection(X_train, X_test, y_train, y_test,f, svclassifier)
    X=X1[features]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(svclassifier, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(svclassifier, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(svclassifier, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(svclassifier, X, y, cv=cv,scoring='f1_micro')
    print("SVM recall",rec)
    print("SVM Precision",pre)
    print("SVM F1",f1)
    print('SVM Accuracy', scores)
    #aucsvm=cross_val_score(svclassifier, X, y, cv=cv,scoring='roc_auc_ovo')
    #print("SVm AUC",aucsvm)
    cvsvm=scores
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    nested_score = cross_val_score(svclassifier, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print('matthews_corrcoef',nested_score)
    #print(forward_feature_selection(X_train, X_test, y_train, y_test,10, svclassifier))
    #resultssvm.append(accuracy_score(y_test, y_pred))
    cvsvm2=nested_score
    '''

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(300,100), max_iter=1000,alpha=0.0001, learning_rate_init=0.01,activation='logistic')
    #X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    #features=forward_feature_selection(X_train, X_test, y_train, y_test,10, mlp)
    #X=X1[features]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(mlp, X, y, cv=cv,scoring='accuracy')
    rec=cross_val_score(mlp, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(mlp, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(mlp, X, y, cv=cv,scoring='f1_micro')
    print("NN recall",rec)
    print("NN Precision",pre)
    print("NN F1",f1)
    print('NN Accuracy', scores)
    aucnn=cross_val_score(mlp, X, y, cv=cv,scoring='roc_auc_ovo')
    print("NN  AUC",aucnn)
    cvmlp=scores
    nested_score = cross_val_score(mlp, X=X, y=y, cv=cv, scoring=make_scorer(matthews_corrcoef))
    print('matthews_corrcoef',nested_score)
    #X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    #print(forward_feature_selection(X_train, X_test, y_train, y_test,10, mlp))
    #resultsmlp.append(accuracy_score(y_test, y_pred))
    cvmlp1=aucnn
    cvmlp2=nested_score




    '''
    def svc_param_selection(X, y, nfolds):
        Cs = [0.0001,0.002, 0.001,0.02, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        kernels=['rbf','sigmoid','linear','poly']
        param_grid = {'C': Cs, 'gamma': gammas, 'kernel':kernels}
        grid_search = GridSearchCV(svm.SVC(kernel=kernels), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_

    print(svc_param_selection(X, y, folds))
    
    svclassifier = SVC(kernel='rbf', random_state=0, gamma=2, C=0.0001)
    # Train the classifier
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    features=forward_feature_selection(X_train, X_test, y_train, y_test,f, svclassifier)
    X=X1[features]
    cv = ShuffleSplit(n_splits=folds, random_state=0)
    scores = cross_val_score(svclassifier, X, y, cv=cv,scoring='accuracy')
    print('SVM RBF Accuracy ', scores)
    rec=cross_val_score(svclassifier, X, y, cv=cv,scoring='recall_micro')
    pre=cross_val_score(svclassifier, X, y, cv=cv,scoring='precision_micro')
    f1=cross_val_score(svclassifier, X, y, cv=cv,scoring='f1_micro')
    print("SVM RBFrecall",rec)
    print("SVM RBF Precision",pre)
    print("SVM RBF F1",f1)
    y_pred = cross_val_predict(svclassifier, X, y, cv=3)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    nested_score = cross_val_score(svclassifier, X=X, y=y, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score))
    print(nested_score)
    X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.1, random_state=42)
    #print(forward_feature_selection(X_train, X_test, y_train, y_test,10, svclassifier))
    
    '''







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
#summarize_results(cvsvm1)
summarize_results(cvsvm2)


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











