import pandas as pd
import numpy as np

def BackTest(data, pred, window,p, q, initialpos, lev, code, HR, name1, name2):
    m = len(pred)
    capital = p[0] / lev
    pricesB = np.full(m, np.nan)
    pricesS = np.full(m, np.nan)
    pos = initialpos
    net = []
    profit = []
    k = 0
    loc = []
    data1 = data.set_index(pd.Series(range(data.shape[0])))
    for t in range(0, m):
        expire = data1[data1[name1] == code[t]][name2]       
        location1 = expire.index.values[-1] 
        expire = pd.to_datetime(expire[location1]).strftime('%Y-%m-%d')
        expire = np.datetime64(expire)   
        if data.iloc[location1:location1+1,:].index == expire:
            location2 = location1 - window
        else:
            location2 = -1
        loc = np.append(loc,location2)
    loc = list(set(loc))
    loc1 = []
    for i in range(0, m):
        if pos == 1:  # RQ>-?U?*
            if pred[i] == 1:
                if i in loc:
                    k = k+1
                    pos = 1
                    capital = capital + (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + (1 - 0.00002) * p[i])*HR[k]
                    gain = (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + (1 - 0.00002) * p[i])*HR[k] - 0.00002*q[i]
                    pricesB[i] = q[i]
                    p[i:i+window+1] = q[i:i+window+1].copy()
                    capital = capital - 0.00002*q[i]
                    net = np.append(net, capital)
                    profit = np.append(profit, gain)
                    loc1.append(i)
                    

                else:
                    pos = 1
                    net = np.append(net, (capital + (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + p[i])*HR[k]))
                    

            elif pred[i] == 0:
                capital = capital + (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + (1 - 0.00002) * p[i])*HR[k]
                pos = 0
                net = np.append(net, capital)
                                

            elif pred[i] == -1:
                if i in loc:
                    pos = -1
                    capital = capital + (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + (1 - 0.00002) * p[i])*HR[k]
                    gain = (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + (1 - 0.00002) * p[i])*HR[k] - 0.00002 * q[i]
                    pricesS[i] = q[i]
                    p[i:i+window+1] = q[i:i+window+1].copy()
                    capital = capital - 0.00002 * pricesS[i]
                    net = np.append(net, capital)
                    profit = np.append(profit, gain)
                    loc1.append(i)
                    

                else:
                    pos = -1
                    capital = capital + (- pricesB[np.max(np.where(~np.isnan(pricesB)))] + (1 - 0.00002) * p[i])*HR[k]
                    pricesS[i] = p[i]
                    capital = capital - 0.00002 * pricesS[i]
                    net = np.append(net, capital)
                                        

        elif pos == -1:  # RQ>-?U?*
            if pred[i] == -1:
                if i in loc:
                    k = k+1
                    pos = -1
                    capital = capital + (pricesS[np.max(np.where(~np.isnan(pricesS)))] - (1 + 0.00002) * p[i])*HR[k]
                    gain = (pricesS[np.max(np.where(~np.isnan(pricesS)))] - (1 + 0.00002) * p[i])*HR[k] - 0.00002 * q[i]
                    pricesS[i] = q[i]
                    p[i:i+window+1] = q[i:i+window+1].copy()
                    capital = capital - 0.00002*q[i]
                    net = np.append(net, capital)
                    profit = np.append(profit, gain)
                    loc1.append(i)
                                        
                else:
                    pos = -1
                    net = np.append(net, (capital + (pricesS[np.max(np.where(~np.isnan(pricesS)))] - p[i])*HR[k]))
                    

            elif pred[i] == 0:
                capital = capital + (pricesS[np.max(np.where(~np.isnan(pricesS)))] - (1 + 0.00002) * p[i])*HR[k]
                pos = 0
                net = np.append(net, capital)
                
            elif pred[i] == 1:
                if i in loc:
                    pos = 1
                    capital = capital + (pricesS[np.max(np.where(~np.isnan(pricesS)))] - (1 + 0.00002) * p[i])*HR[k]
                    gain = (pricesS[np.max(np.where(~np.isnan(pricesS)))] - (1 + 0.00002) * p[i])*HR[k] - 0.00002 * q[i]
                    pricesB[i] = q[i]
                    p[i:i+window+1] = q[i:i+window+1].copy()
                    capital = capital - 0.00002 * pricesB[i]
                    net = np.append(net, capital)
                    profit = np.append(profit, gain)
                    loc1.append(i)
                    
                else:
                    pos = 1
                    capital = capital + (pricesS[np.max(np.where(~np.isnan(pricesS)))] - (1 + 0.00002) * p[i])*HR[k]
                    pricesB[i] = p[i]
                    capital = capital - 0.00002 * pricesB[i]
                    net = np.append(net, capital)
                    
        elif pos == 0:  # RQ>-F=2V
            if pred[i] == 1:
                pricesB[i] = p[i]
                pos = 1
                capital = capital - 0.00002 * pricesB[i]
                net = np.append(net, capital)
                
                
            elif pred[i] == -1:
                pricesS[i] = p[i]
                pos = -1
                capital = capital - 0.00002 * pricesS[i]
                net = np.append(net, capital)
                
                
            else:
                net = np.append(net, capital)
                                

    return net, profit, loc1
