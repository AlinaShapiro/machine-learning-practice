import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def calc_metrics(true, pred):
    mse = metrics.mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(true, pred)
    r2 = metrics.r2_score(true,
                          pred)

    print("RMSE:  {}\nMAE:   {} \nR2:   {}".format(rmse, mae, r2))


if __name__ == '__main__':

    df = pd.read_csv('insurance.csv')
    df['const'] = 1
    y = df['charges']
    x = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    data_X = np.asarray(x)
    data_Y = np.asarray(y)

    for i in range(0, len(data_X)):
        if data_X[i][1] == 'female':
            data_X[i][1] = 0
        elif data_X[i][1] == 'male':
            data_X[i][1] = 1
        if data_X[i][4] == 'yes':
            data_X[i][4] = 1
        elif data_X[i][4] == 'no':
            data_X[i][4] = 0

        if data_X[i][5] == 'southwest':
            data_X[i][5] = 1
        elif data_X[i][5] == 'southeast':
            data_X[i][5] = 2
        elif data_X[i][5] == 'northwest':
            data_X[i][5] = 3
        elif data_X[i][5] == 'northeast':
            data_X[i][5] = 4

    print("\ndata_X: ", data_X)
    print("\ndata_y: ", data_Y)

    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=1)

    print("Классификатор LinearRegression")
    model_linear = LinearRegression()
    model_linear.fit(X_train, Y_train)
    print(model_linear.coef_, model_linear.intercept_)
    y_pred_linear = model_linear.predict(X_test)
    calc_metrics(Y_test, y_pred_linear)
    print("\n")

    print("Классификатор Lasso")
    model_linear = Lasso()
    model_linear.fit(X_train, Y_train)
    print(model_linear.coef_, model_linear.intercept_)
    y_pred_linear = model_linear.predict(X_test)
    calc_metrics(Y_test, y_pred_linear)
    print("\n")

    print("Классификатор Ridge")
    model_linear = Ridge()
    model_linear.fit(X_train, Y_train)
    print(model_linear.coef_, model_linear.intercept_)
    y_pred_linear = model_linear.predict(X_test)
    calc_metrics(Y_test, y_pred_linear)
    print("\n")


