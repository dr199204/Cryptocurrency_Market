import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from statsmodels.tsa.stattools import adfuller


###function to scrape data from coinmarket
def cb_hist_data_scrap(coin,startdate,enddate):
    coin=coin.lower().replace(" ","-")
    r  = requests.get("https://coinmarketcap.com/currencies/{0}/historical-data/?start={1}&end={2}".format(coin, startdate, enddate))
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    tables = soup.findAll("table")
    table=tables[2]
    df=pd.read_html(str(table),parse_dates=True)[0]
    df['name']=coin
    df[['Market_Cap_in_bn','Volume_in_bn']]=df[['Market Cap','Volume']].div(1e9)
    df.rename(columns={"Open*": "Open", "Close**": "Close"},inplace=True)
    df['Date']=pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    df.sort_index(inplace=True)
    return df
    
def summary_stats_partition (input_list, n):
    df_sum_stats=pd.DataFrame(index=list(range(0,n)),columns=['Mean','StDev'])
    parts=[input_list[i::n].tolist() for i in range(n)]
    for p in parts:
        df_sum_stats.at[parts.index(p),'Mean']=np.mean(p)
        df_sum_stats.at[parts.index(p),'StDev']=np.std(p)
    return df_sum_stats
    
def is_stationary(r):
    ad_fuller=adfuller(r)
    print('ADF Statistic: %f'%ad_fuller[0])
    print('p-value: %f'%ad_fuller[1])
    pvalue=ad_fuller[1]
    for key,value in ad_fuller[4].items():
         if ad_fuller[0]>value:
            print("The series is not stationary.")
            break
         else:
            print("The series is stationary.")
            break;
    print('Critical values:')
    for key,value in ad_fuller[4].items():
        print('\t%s: %.3f ' % (key, value))
        
        ###Volatility
def skewness(r,ddof=0):
    """
    Computes the skewness of Series or DataFrame
    Returns a float or a Series
    Parameters:
            r (Series of DataFrame): returns
            ddof (int): degrees of freedom. Default is 0.
    """
    if not isinstance(r,(pd.Series,pd.DataFrame)):
        raise Exception("Please input a Series or a DataFrame")
    if not ddof in [0,1]:
        raise Exception("Please input a valid value for ddof")
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=ddof)
    exp = (demeaned_r**3).mean()
    if ddof==1:
        print("Calculating std with Bessel's correction")
    skewness=exp/sigma_r**3
    print('Skewness: %f'%skewness)
    if skewness<0:
        print("Distribution is left skewed")
    elif skewness>0:
        print("Distribution is right skewed")
    return skewness
    


def kurtosis(r,ddof=0):
    """
    Computes the kurtosis of the Series or DataFrame
    Returns a float or a Series
    """
    if not isinstance(r,(pd.Series,pd.DataFrame)):
        raise Exception("Please input a Series or a DataFrame")
    if not ddof in [0,1]:
        raise Exception("Please input a valid value for ddof")
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    if ddof==1:
        print("Calculating std with Bessel's correction")
    kurtosis=exp/sigma_r**4
    print('Kurtosis: %f'%kurtosis)
    if kurtosis<3:
        print("Distribution is platykurtic")
    elif kurtosis>3:
        print("Distribution is leptokurtic")
    return kurtosis
    
    
def jb_test_is_normal(r, critical_level=0.01):
    from scipy.stats import jarque_bera
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = jarque_bera(r)
        print('J_B Statistic: %f'%statistic)
        print('p-value: %f'%p_value)
        if p_value>critical_level:
            print("The distribution is normal.")
        else:
            print("The series is not normal.") 
