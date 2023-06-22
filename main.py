import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet, mobilenet_v2


def calc_metrics_cls(true, pred):
    stats = {
        'Accuracy': round(metrics.accuracy_score(true, pred), 2),
        'ROC AUC': round(metrics.roc_auc_score(true, pred), 2),
        'F1': round(metrics.f1_score(true, pred), 2)
    }
    print(f"Accuracy: {stats['Accuracy']}\nROC AUC: {stats['ROC AUC']}\nF1: {stats['F1']}")
    return stats

if __name__ == '__main__':
    dir = './brain/'

    yes_dir = os.path.join(dir, 'yes')
    fnames = [os.path.join(yes_dir, fname) for fname in os.listdir(yes_dir)][:]
    print(len(fnames))
    no_dir = os.path.join(dir, 'no')
    fnames.extend([os.path.join(no_dir, fname) for fname in os.listdir(no_dir)][:])
    print(len(fnames))
    print(fnames)  # пути к каждой картинке

    X = []
    INPUT_SHAPE = 224
    net_model = mobilenet_v2
    net_model_class = net_model.MobileNetV2
    c = 0
    # Features
    model = net_model_class(weights='imagenet', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3), include_top=False,
                            pooling='avg')

    features_dict = {}
    for img_path in fnames:
        c += 1
        # print(img_path)
        img = image.load_img(img_path, target_size=(INPUT_SHAPE, INPUT_SHAPE))  # делаем определенный размер
        x = image.img_to_array(img)  # Преобразует экземпляр PIL Image в массив Numpy.
        # print(x)
        x = np.expand_dims(x, axis=0)
        # print(x)
        x = net_model.preprocess_input(x)  # преобразует аргументы в нужный формат

        preds = model.predict(x)
        features_dict[str(os.path.basename(img_path) + "_" + str(c))] = preds
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        X.append(np.array(preds))
        print('Predicted:', preds.shape, preds)
    # print("features_dict",features_dict)
    print(X)
    print(c)

    print(features_dict.items())

    Y = []
    count_pos = 0
    count_neg = 0
    for img_path in fnames:
        if 'Y' in os.path.basename(img_path):
            Y.append(1)
            count_pos += 1
        else:
            Y.append(0)
            count_neg += 1
    print(Y)
    print('count_pos: ', count_pos, '\ncount_neg: ', count_neg)

    X = np.reshape(np.array(X), (253, 1280))
    print(X)
    print(X.shape)

    df = pd.DataFrame(data=X)
    df['Y'] = Y
    print(df.head())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    cls_models_stats = {}

    print("DecisionTreeClassifier")
    cls_model_dt = DecisionTreeClassifier()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['DecisionTree'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print("SVC")
    cls_model_svc = SVC()
    cls_model_svc.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_svc.predict(X_test)
    cls_models_stats['SVC'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print("RandomForest")
    cls_model_dt = RandomForestClassifier()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['RandomForest'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print("SGDClassifier")
    cls_model_dt = SGDClassifier()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['SGDClassifier'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print("LogisticRegression")
    cls_model_dt = LogisticRegression()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['LogisticRegression'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    print("GaussianNB")
    cls_model_dt = GaussianNB()
    cls_model_dt.fit(X_train, Y_train)
    y_pred_cls_dt = cls_model_dt.predict(X_test)
    cls_models_stats['GaussianNB'] = calc_metrics_cls(Y_test, y_pred_cls_dt)

    model_cls_compare = pd.DataFrame(cls_models_stats).T
    print(model_cls_compare)







