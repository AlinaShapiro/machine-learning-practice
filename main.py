import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def calc_metrics_cls(true, pred):
    stats = {
        'Accuracy': round(metrics.accuracy_score(true, pred), 2),
        'ROC AUC': round(metrics.roc_auc_score(true, pred), 2),
        'F1': round(metrics.f1_score(true, pred), 2)
    }
    print(f"Accuracy: {stats['Accuracy']}\nROC AUC: {stats['ROC AUC']}\nF1: {stats['F1']}")
    return stats

if __name__ == '__main__':
    df = pd.read_csv(r"./datasets/voice.csv", header=0, sep=',')
    df.head()

    X = df[['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm',
            'mode', 'centroid', 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom',
            'dfrange', 'modindx']]
    df.loc[df['label'] == 'female', 'label'] = 1
    df.loc[df['label'] == 'male', 'label'] = 0
    Y = df['label']
    Y = Y.astype({'label': np.int})

    data_X = np.array(X)
    data_Y = np.array(Y)
    print(data_X)
    print(data_Y)

    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=1)

    cls_models_stats = {}

    print('DecisionTree')
    cls_model_dt = DecisionTreeClassifier()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['DecisionTree'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print('SVC')
    cls_model_svc = SVC()
    cls_model_svc.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_svc.predict(X_test)
    cls_models_stats['SVC'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print('RandomForest')
    cls_model_dt = RandomForestClassifier()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['RandomForest'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print('SGDClassifier')
    cls_model_dt = SGDClassifier()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['SGDClassifier'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print('LogisticRegression')
    cls_model_dt = LogisticRegression()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['LogisticRegression'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print('GaussianNB')
    cls_model_dt = GaussianNB()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['GaussianNB'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    model_cls_compare = pd.DataFrame(cls_models_stats).T
    print(model_cls_compare)


