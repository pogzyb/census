"""
imbalanced dataset - UCI Census Income Dataset
http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/
# exploration of Precision vs. Recall metrics using
# - PCA
# - undersampling and oversampling
# - SMOTE
# - LogisticRegression, RandomForestClassifier, and SupportVectorMachine

Author: Joe Obarzanek -
"""
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split,
                                    GridSearchCV,
                                    StratifiedKFold)
import sklearn.metrics as metrics
import sklearn.ensemble as skens
import sklearn.tree as tree
import sklearn.linear_model as sklinear
import sklearn.pipeline as skpipe
from sklearn import svm

import warnings
warnings.filterwarnings("ignore")

# - - -

# class for initial data cleaning and prep
class CleanandPrep(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        # transformations
            # - convert nomials to str for dummifying
            # - drop 'instance weight'
        nomials = ['own business or self employed','veterans benefits','year']
        X[nomials] = X[nomials].astype("object")
        X = X.drop('instance weight', axis=1)
        return X

    def fit(self, X, y=None, **kwargs):
        return self

# class for dummifying / onehotencoding categorical variables
class DummyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.cols = []
        pass

    def transform(self, X, y=None, **kwargs):
        X = pd.get_dummies(X)
        # set training columns
        if len(self.cols) == 0:
            self.cols = X.columns
        else:
            # check for mismatched columns: test split might have had a value
            # or two that did not appear in the training split (and vice-versa).
            # columns: "detailed household and family stat" and "country of birth self"
            # seem to be the main culprits
            missingTest = [x for x in self.cols if x not in X.columns]
            missingTrain = [x for x in X.columns if x not in self.cols]
            print("\n Mismatched columns - fixing now. . .\n")
            if len(missingTest) > 0:
                for col in missingTest:
                    X[col] = 0
                    print("added column to Test: {} - that was in Train".format(col))
            if len(missingTrain) > 0:
                for col in missingTrain:
                    del X[col]
                    print("deleted column in Test: {} - that was not in Train".format(col))
        return X

    def fit(self, X, y=None, **kwargs):
        return self

# class for numeric varible scaling - StandardScaler() can be changed out
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        X = self.scaler.transform(X)
        return X

    def fit(self, X, y=None, **fit_params):
        self.scaler = StandardScaler().fit(X)
        return self
# - - -

def scoringReport(y_true, y_pred):
    roc_auc = metrics.roc_auc_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)
    print("ROC_AUC Score: %s" %roc_auc)
    print("Confusion Matrix:\n%s\n" %cm)
    print(metrics.classification_report(y_true, y_pred))


def samplingMethods(df, target, over=False, under=False, seed=33):
    # not truly oversampling for cpu memory sake, but close
    if over:
        print("Original data shape: {}".format(df.shape))
        pos = df.loc[df[target] == 1]
        neg = df.loc[df[target] == 0]
        # resample
        pos = pos.sample(n=int(len(neg)/2), replace=True, random_state=seed)
        neg = neg.sample(n=int(len(neg)/2), replace=False, random_state=seed)
        df = pd.concat([pos,neg], ignore_index=True)
        print("Oversampled data shape: {}".format(df.shape))
        return df
    # undersampling
    elif under:
        print("Original data shape: {}".format(df.shape))
        pos = df.loc[df[target] == 1]
        neg = df.loc[df[target] == 0]
        # resample
        neg = neg.sample(len(pos), replace=False, random_state=seed)
        df = pd.concat([pos,neg], ignore_index=True)
        print("Undersampled data shape: {}".format(df.shape))
        return df
    return "Error - No Method Specified - Check Params."

# - - -

def main():
    # read files
    training = pd.read_csv('training.csv', na_values=['?'])
    testing = pd.read_csv('testing.csv', na_values=['?'])
    # confirmation + data shapes
    print("Training data:\n Rows: %s \t Columns: %s" %(training.shape[0], training.shape[1]))
    print("Testing data:\n Rows: %s \t Columns: %s" %(testing.shape[0], testing.shape[1]))
    # convert targets '+50000' / '-50000' -> 1 / 0
    training.target = training.target.apply(lambda x: int(x.strip('+.\n').replace(' ','')))
    training.target = training.target.map({-50000:0,50000:1})

    testing.target = testing.target.apply(lambda x: int(x.strip('+.\n').replace(' ','')))
    testing.target = testing.target.map({-50000:0,50000:1})

    # target
    print("Training target dist:\n%s" %training.target.value_counts())
    print("Testing target dist:\n%s" %testing.target.value_counts())
    # Select sampling method
    training = samplingMethods(training, "target", under=True)

    # -- -- --
    # X y splits
    X = training.loc[:,training.columns != "target"]
    y = training["target"].values
    # separate training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y,
                                                test_size=0.3, random_state=12)

    # testing data splits
    X_test = testing.loc[:,testing.columns!='target']
    y_test = testing["target"]

    # stratified cross validation
    # - very useful for imbalanced datasets
    kf = StratifiedKFold(3, shuffle=True, random_state=33)

    # make pipeline
    #  - PCA optional (cuts runtime for randomforests and other algos)
    pipe = skpipe.make_pipeline(CleanandPrep(),
                                DummyEncoder(),
                                Scaler(),
                                PCA(n_components=170, random_state=33),
                                sklinear.LogisticRegression(C=0.7, random_state=33)
                                # svm.SVC()
                                )

    # initial trials of pipeline by itself w/o GridSearchCV
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    scoringReport(y_valid, preds)

    # LogReg Params
    cv_params = {
        "logisticregression__class_weight":[None,{1:10,0:1}], #[None,{1:10,0:1}]
        "logisticregression__penalty":['l1'] #['l2','l1']
    }
    #
    grid = GridSearchCV(pipe, param_grid=cv_params, scoring='roc_auc', cv=kf)
    grid.fit(X_train, y_train)
    print("Training .fit Summary:\nBest Score: \
     {}\nBest Params: {}\n".format(grid.best_score_, grid.best_params_))
    # validation set (no re-fit)
    print("Validation set .predict Results:\n")
    preds = grid.predict(X_valid)
    scoringReport(y_valid, preds)
    # # test set (re-fit with all training data)
    # grid.fit(X, y)
    # print("Complete Training .fit Summary:\nBest Score: \
    #  {}\nBest Params: {}\n".format(grid.best_score_, grid.best_params_))
    # preds = grid.predict(X_test)
    # print("Test set .predict Results:\n")
    # scoringReport(y_test, preds)

    return

# - - -
if __name__ == "__main__":
    main()
