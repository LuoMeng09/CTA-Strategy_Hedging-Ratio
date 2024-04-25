import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def OLS(df, price, index00, time, window, price00, price01, date00, futures, location):
    df1 = df.copy()
    for t in futures:
        maturity = df1[df1[index00] == t][date00]
        location1 = maturity.index.values[-1]
        location1 = int(np.where(df1.index==location1)[0])
        location2 = location1 - window  
        
        if location2 > 0:
            df1[price00].iloc[location2:location1 + 1] = df1[price01].iloc[location2:location1 + 1]
    result = {}
    for i in futures[location:]:
        maturity = df1[df1[index00] == i][date00]
        location1 = maturity.index.values[-1] 
        location1 = int(np.where(df1.index==location1)[0])
        location2 = location1 - window  
        data = df1.iloc[location2 - time:location2 + 1, :]
        data = data.set_index(pd.Series(range(data.shape[0])))

        Asset = data[price].astype('float') 
        Future = data[price00].astype('float') 
        data['Asset1'] = (Asset.shift(-1) - Asset).fillna(0)
        data['Future1'] = (Future.shift(-1) - Future).fillna(0)

        X = np.array(data['Future1']).reshape((-1, 1))
        y = np.array(data['Asset1'])
        model0 = LinearRegression().fit(X, y)
        HR = model0.coef_
        Var_P = np.std(data['Asset1'] - HR * data['Future1'])
        Var_A = np.std(data['Asset1'])
        HE = 1 - Var_P / Var_A
        result[i] = HR

    res = pd.DataFrame(result).T
    res.columns = ['OLS']

    return res