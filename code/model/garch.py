import pandas as pd
import numpy as np
from arch import arch_model

def GARCH(df, index00, time, window, price00, price01, date00, futures, location, p, q):
    df1 = df.copy()
    for t in futures:
        maturity = df1[df1[index00] == t][date00]
        location1 = maturity.index.values[-1]
        location1 = int(np.where(df1.index==location1)[0])        
        location2 = location1 - window
      
        if location2 > 0:
            df1[price00].iloc[location2:location1 + 1] = df1[price01].iloc[location2:location1 + 1]
    result3 = {}
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

        X = np.array(data['Future1']).reshape((-1, 1))
        y = np.array(data['Asset1'])
        am0 = arch_model(y, X, mean='LS', lags=0, vol='GARCH', p=p, o=0, q=q)
        model = am0.fit()
        HR = model.params[1]        
# =============================================================================
#         model0 = am0.fit()
#         aic = model0.aic
#         HR = model0.params[1]
# #        s_optimal = 1
# #        t_optimal = 1
#         for s in range(1, a + 1):
#             for t in range(1, b + 1):
#                 am = arch_model(y, X, mean='LS', lags=0, vol='GARCH', p=p, o=0, q=q)
#                 model = am.fit()
#                 if model.aic < aic:
#                     aic = model.aic
#                     HR = model.params[1]
#                     s_optimal = s
#                     t_optimal = t
# =============================================================================
        result3[i] = HR

# =============================================================================
#     am = arch_model(y, X, mean='LS', lags=0, vol='GARCH', p=s_optimal, o=0, q=t_optimal).fit()
#     print('p=%d'%s_optimal ,'q=%d'%t_optimal)
#     print(am.summary())
# =============================================================================
    res3 = pd.DataFrame([result3]).T
    res3.columns = ['GARCH']
    return res3