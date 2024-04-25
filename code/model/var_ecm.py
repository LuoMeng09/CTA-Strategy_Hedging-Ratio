from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

def VAR(df, index00, time, window, price00, price01, date00, futures, location, varLagNum):
    df1 = df.copy()
    for t in futures:
        maturity = df1[df1[index00] == t][date00]
        location1 = maturity.index.values[-1]
        location1 = int(np.where(df1.index==location1)[0])
        location2 = location1 - window 
        if location2 > 0:
            df1[price00].iloc[location2:location1 + 1] = df1[price01].iloc[location2:location1 + 1]
    result1 = {}
    result2 = {}
    for i in futures[location:]:
        maturity = df1[df1[index00] == i][date00]
        location1 = maturity.index.values[-1]
        location1 = int(np.where(df1.index==location1)[0])
        location2 = location1 - window

        data = df1.iloc[location2 - time:location2 + 1, :]
        data = data.set_index(pd.Series(range(data.shape[0])))

        Asset = data[price].astype('float')  # 现货价格
        Future = data[price00].astype('float')  # 期货价格
        data['Asset1'] = (Asset.shift(-1) - Asset).fillna(0)
        data['Future1'] = (Future.shift(-1) - Future).fillna(0)

        # VAR
        adfuller(data['Asset1'])
        adfuller(data['Future1'])
        coint(data['Asset1'], data['Future1'])

        model = sm.tsa.VAR(data[['Future1', 'Asset1']])
        res = model.fit(1)
        aic = res.aic
        optimalNum = 1
        for j in range(2, varLagNum + 1):
            ml = model.fit(j)
            if ml.aic < aic:
                aic = ml.aic
                optimalNum = j

        res = model.fit(optimalNum)
        params = res.params
        Future1_pred = params.iloc[0, 0]
        Asset1_pred = params.iloc[0, 1]
        for m in range(optimalNum):
            Future1_pred = Future1_pred + params.iloc[2 * m + 1, 0] * data['Future1'].shift(m + 1)
            Future1_pred = Future1_pred + params.iloc[2 * m + 2, 0] * data['Asset1'].shift(m + 1)
            Asset1_pred = Asset1_pred + params.iloc[2 * m + 1, 1] * data['Future1'].shift(m + 1)
            Asset1_pred = Asset1_pred + params.iloc[2 * m + 2, 1] * data['Asset1'].shift(m + 1)

        error1 = Future1_pred - data['Future1']
        error11 = error1.dropna()
        error2 = Asset1_pred - data['Asset1']
        error22 = error2.dropna()
        sigma1 = np.std(error11)
        sigma2 = np.std(error22)
        cor = error11.corr(error22)
        HR1 = cor * sigma2 / sigma1
        Var_P1 = np.std(data['Asset1'] - HR1 * data['Future1'])
        Var_A1 = np.std(data['Asset1'])
        HE1 = 1 - Var_P1 / Var_A1
        result1[i] = HR1
        # ECM
        # data1 = error1.shift(1)
        data1 = (Asset - Future).shift(1)
        for p in range(optimalNum):
            data1 = pd.concat([data1, data['Future1'].shift(p + 1), data['Asset1'].shift(p + 1)], axis=1)
        X1 = data1.dropna(axis=0)
        y1 = data['Future1'][data1.shape[0] - X1.shape[0]:]
        model1 = LinearRegression().fit(X1, y1)
        # print(model1.coef_)
        error111 = model1.predict(X1) - y1

        # data2 = error2.shift(1)
        data2 = (Asset - Future).shift(1)
        for q in range(varLagNum):
            data2 = pd.concat([data2, data['Future1'].shift(q + 1), data['Asset1'].shift(q + 1)], axis=1)
        X2 = data2.dropna(axis=0)
        y2 = data['Asset1'][data2.shape[0] - X2.shape[0]:]
        model2 = LinearRegression().fit(X2, y2)
        # print(model2.coef_)
        error222 = model2.predict(X2) - y2

        HR2 = error111.corr(error222) * np.std(error222) / np.std(error111)
        Var_P2 = np.std(data['Asset1'] - HR2 * data['Future1'])
        Var_A2 = np.std(data['Asset1'])
        HE2 = 1 - Var_P2 / Var_A2
        result2[i] = HR2

    res1 = pd.DataFrame([result1]).T
    res2 = pd.DataFrame([result2]).T
    res1.columns = ['VAR']
    res2.columns = ['ECM']

    print('optimalNum=%d'%optimalNum)
    print(res.summary())
    return res1, res2
