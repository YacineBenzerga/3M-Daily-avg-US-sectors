import numpy as np
import pandas as pd

raw = pd.read_excel('Financial Data.xlsx')
raw.reset_index(inplace=True)
raw['Symbol'] = raw['Symbol'].apply(lambda x:x.split(' ')[0])
raw.set_index('Symbol', inplace=True)
raw.drop('index' , axis=1 ,inplace=True)


rel = ['PricetoBookValue',
 'Gross Profit(Quart)',
 'Return on Equity (TTM)',
 'Fixed Asset Turnover(Quartl)',
 'Earning Yield(forward 1y)',
 'Free Cash Flow Yield(TTM)',
 '30 D Avg Daily Volume',
 'Momentum Score',
 'Market Cap',
 'Total Returns Price',
 '3M Price Returns (Daily)']

for c in rel:
    raw[c] = raw[c].astype(str)
    
to_fix_columns = ['Gross Profit(Quart)' , '30 D Avg Daily Volume' , 'Market Cap']
numeric_columns = ['PricetoBookValue' ,'Return on Equity (TTM)' ,'Fixed Asset Turnover(Quartl)' ,'Earning Yield(forward 1y)' ,'Free Cash Flow Yield(TTM)' ,'Momentum Score' ,'Total Returns Price','3M Price Returns (Daily)']
for col in raw[to_fix_columns]:
    raw[col] = raw[col].apply(lambda x:x.replace('.','').replace('B','0'))
    raw[col] = raw[col].apply(lambda x:x.replace('M',''))

for c in rel:
    raw[c] = raw[c].astype('float64')
    
    
    
anomalie_yield =list(raw[raw['Earning Yield(forward 1y)']>150].index)
anomalie_return = list(raw[raw['Total Returns Price']>1500].index)
anomalie_Cash_flow = list(raw[raw['Free Cash Flow Yield(TTM)']<-1.5].index)
anomalie_market_cap = list(raw[raw['Market Cap']>400000].index)

anomalies_list = anomalie_yield+anomalie_Cash_flow + anomalie_market_cap + anomalie_return
anomalies_data = pd.DataFrame(raw,index=anomalies_list)
raw.drop(anomalies_list, axis=0 , inplace=True)
anomalies_data.drop(['THC'] , axis=0,inplace=True)

missing_values = pd.DataFrame(raw.isnull().sum())
missing_values.columns=['Missing']
missing_values=missing_values.sort_values(by='Missing' ,ascending=False)
for col in raw[rel]:
    raw[col].fillna(raw[col].median() ,inplace=True)
    
raw.to_csv('raw_cleaned.csv',index=True)