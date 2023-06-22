import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #1. Сформировать новый файл, содержащий статистику по заболеванию covid-19, в котором содержится информация только за последние 7 дней.
    dataX = pd.read_csv(r"./time_series_covid19_confirmed_global.csv", header=0, sep=',')
    dataX.head()

    dataX_new_1 = dataX.loc[:,['Country/Region', '11/14/22', '11/15/22', '11/16/22', '11/17/22', '11/18/22', '11/19/22','11/20/22']]
    dataX_new_1.to_csv('./time_series_covid19_confirmed_last_7_days.csv', sep=',', header=True)

    dataX_new_2 = dataX.loc[:, ['Country/Region', '11/20/22']]
    dataX_new_2.to_csv('./time_series_covid19_confirmed_last__days.csv', sep=',', header=True)

    #2 . Сформировать файл ( с данными за последний день), в котором есть страны с числом подтвержденных случаев COVID-19
    # большим некоторого числа (порог = 10 000 000).
    frame = dataX.loc[:, ['Country/Region', '11/20/22']]
    frame.rename(columns={'Country/Region': 'Country', '11/20/22': 'day112022'}, inplace=True)
    print(frame)
    frame.to_csv('./time_series_covid19_confirmed_last__day_more_10000000.csv', sep=',', header=True)
    frame[(frame.day112022.astype(int) >= 10000000)].to_csv('./time_series_covid19_confirmed_last__day_more_10000000.csv', sep=',', header=True)
    print(frame[(frame.day112022.astype(int) >= 10000000)])

    #3. Нарисовать диаграмму с данными из пункта 2.

    frame = dataX.loc[:, ['Country/Region', '11/20/22']]
    y = frame.columns
    Xframe = frame[frame[y[1]].astype(int) > 10000000]
    fig, ax = plt.subplots()
    ax.axis("equal")
    vals = Xframe[y[1]]  # числовые значения
    labels = Xframe[y[0]]  # название секторов
    ax.pie(vals, labels=labels, radius=1.5, autopct='%.1f')
    plt.show()

    frame = dataX.loc[:, ['Country/Region', '11/20/22']]
    y = frame.columns
    highest = frame[(frame[y[1]].astype(int) > 10000000)]
    highest.plot(kind='bar', x=0, y=1, color='maroon')
    plt.show()

    #4. Найти максимальное число подтвержденных случаев COVID-19 и страну за последний месяц.

    print("Максимальное число подтвержденных случаев COVID-19 и страну за последний месяц:")
    dataX = pd.read_csv(r"./time_series_covid19_confirmed_global.csv", header=0,
                        sep=',')
    # dataX.head()
    the_last_month = dataX.loc[:,
                     ['Country/Region', '10/22/22', '10/23/22', '10/24/22', '10/25/22', '10/26/22', '10/27/22',
                      '10/28/22', '10/29/22', '10/30/22', '10/31/22', '11/1/22', '11/2/22',
                      '11/3/22', '11/4/22', '11/5/22', '11/6/22', '11/7/22', '11/8/22',
                      '11/9/22', '11/10/22', '11/11/22', '11/12/22', '11/13/22', '11/14/22',
                      '11/15/22', '11/16/22', '11/17/22', '11/18/22', '11/19/22', '11/20/22']]
    y = the_last_month.columns
    # print(y)
    # print(the_last_month)
    max_date = ''
    max_number = 0
    max_country = ''
    for i in range(0, 289):
        for j in range(1, 31):
            if the_last_month.iloc[i][j] >= max_number:
                max_number = the_last_month.iloc[i][j]
                max_date = y[j]
                max_country = the_last_month.iloc[i][0]

    # print(i, '   ', j)
    print("Date: ", max_date)
    print("Country: ", max_country)
    print("Number: ", max_number)

    # 5. Найти среднее число подтвержденных случаев COVID-19 за неделю для топ-10 стран
    dataX = pd.read_csv(r"./time_series_covid19_confirmed_global.csv", header=0,
                        sep=',')
    dataX_new_1 = dataX.loc[:,
                  ['Country/Region', '11/14/22', '11/15/22', '11/16/22', '11/17/22', '11/18/22', '11/19/22',
                   '11/20/22']]
    dataX_new_1['sum'] = dataX_new_1.sum(axis=1)


    count = 0
    n = 289
    max_index = []
    print('Top 10 avarage:')
    while count < 10:
        row = 0
        max_number = 0
        max_country = ''
        for i in range(0, n):
            if i in max_index:
                continue
            for j in range(1, 8):
                if dataX_new_1.iloc[i, j] >= max_number:
                    row = i
                    max_number = dataX_new_1.iat[i, j]
                    max_country = dataX_new_1.iat[i, 0]
        print(max_country, ": ", round(dataX_new_1.iat[row, -1] / 7))
        max_index.append(row)
        n -= 1
        count += 1

