import os
from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
yf.pdr_override()
import numpy as np
import math

itemtypes = {
    'Item1': 'Business',
    'Item1A': 'Risk Factors',
    'Item3': 'Legal Proceedings',
    'Item7': 'MDA_FCRO',
    'Item7A': 'QQD_MR'
}
def getfilelist(path):
    lst1 = []
    for root, dirs, files in os.walk(os.path.join(path)):
        for file in files:
            if file.endswith(".txt"):
                lst1.append(os.path.join(root, file))
    return lst1

def getInfofromfiles(fileslist, companyCIK,start = 2008, end=2020 ):
    fileslist = [i for i in fileslist if str(companyCIK) in i]
    yearDict = {}
    for i in range(start, end+1):
        yearDict[i] = {}
    for file in fileslist:
        if '10KA' in file or '10KT' in file or '10K' not in file:
            continue
        maincontent = file.split('10K')[1]
        year, itemType, _ = maincontent[1:].split('_')
        year = int(year[0:4])
        if year not in range(2008,2021):
            continue
        item = itemtypes[itemType]
        with open(file) as f:
            itemContent = f.read()
            yearDict[year][item]=itemContent
    #print('total 10ka files',count)
    retDF =  pd.DataFrame.from_dict(yearDict, orient='index')
    #retDF['year'] = retDF.index
    retDF.reset_index(level=0, inplace=True)
    retDF = retDF.rename(columns={'index': 'year'})
    return retDF.reset_index(level=0)

def getLabelsfromStocks(stockInfo, stockInfoDJI):
    parsedCompanydf = pd.merge(stockInfoDJI, stockInfo, how='outer', left_index=True, right_index=True)
    info_dict = parsedCompanydf.set_index(parsedCompanydf.index).T.to_dict('list')
    growthlist = [0]
    for i in range(2008, 2020):
        _, dji, _, stock = info_dict[i]
        _, dji_n, _, stock_n = info_dict[i+1]
        if  math.isnan(stock) or math.isnan(stock):
            growthlist.append(math.nan)
            continue
        stock_annual_growth = (stock_n/stock)-1
        dji_annual_growth = (dji_n/dji) - 1
        growth = stock_annual_growth- dji_annual_growth
        growthlist.append(round(growth,2))

    parsedCompanydf['label'] = np.array(growthlist)
    parsedCompanydf.dropna(inplace=True)
    parsedCompanydf['label'] = parsedCompanydf["label"].apply(lambda x: 0 if x < 0 else (2 if x > 0 else 1))
    parsedCompanydf.reset_index(level=0, inplace=True)
    parsedCompanydf = parsedCompanydf.rename(columns={'Date': 'year', 'Ticker_y':'Ticker', 'Close_y':'Close'})
    parsedCompanydf = parsedCompanydf[['year','Ticker', 'Close','label']]
    return parsedCompanydf


def getcompleteDF(lst1):
    cols = ['Name','CIK','year', 'Ticker','Business','Risk Factors','Legal Proceedings','MDA_FCRO','QQD_MR', 'label']
    companyInfoDF = getCompanyInfo()
    listofcompanyDF = []
    for index, row in companyInfoDF.iterrows():
        stockInfo = getStockInfo(row['Ticker'])
        stockInfoDJI = getStockInfoDJI()
        stockInfowithlabels = getLabelsfromStocks(stockInfo, stockInfoDJI)
        parsedCompanydf = getInfofromfiles(filelist, row['CIK'])
        parsedCompanydf = parsedCompanydf.drop('index', 1)
        parsedCompanydf = pd.merge(parsedCompanydf, stockInfowithlabels, on=['year'], how='inner')
        parsedCompanydf['Name'] = row['Name']
        parsedCompanydf['CIK'] = row['CIK']
        parsedCompanydf = parsedCompanydf.drop('Close', 1)
        #parsedCompanydf['label']= stockInfowithlabels['label'].values ##
        listofcompanyDF.append(parsedCompanydf)
    finalDF = pd.concat(listofcompanyDF, ignore_index=True)
    finalDF = finalDF[cols]
    return finalDF

def getCompanyInfo():
    companyDF = pd.read_csv('cik_ticker_list.csv',delimiter='|')
    dowDF = pd.read_csv("../SEC-EDGAR-text/companies_list.txt",sep=" ", names=["CIK", "Ticker"])
    return pd.merge(companyDF, dowDF, on=["CIK","Ticker"])[['Ticker', 'Name','CIK']]

def getStockInfo(company,start="2008-02-01",end="2020-12-31"):
    info = pdr.get_data_yahoo(company, start=start, end=end, interval="1d")
    crit2 = info.index.map(lambda x : x.month == 12 and (x.day==31 or x.day==30 or x.day==29 ))
    info = info[crit2]
    info.index = pd.to_datetime(info.index, format='%Y-%m-%d').year
    info = info.dropna()
    info.groupby(info.index).max()
    info['Ticker'] = company
    df = info[['Ticker','Close']]
    df = df.groupby(df.index).max()
    return df

def getStockInfoDJI():
    df = getStockInfo('DJI')
    df.loc[2019] = ['DJI', 28538.4]
    df = df.sort_index()
    return df

filelist = getfilelist('../SEC-EDGAR-text/output_files_examples/batch_0004')
#companyInfoDF = getCompanyInfo()
#stockInfo = getStockInfo('AAPL')
#stockInfoDJI = getStockInfoDJI()
#compandf = getInfofromfiles(filelist, 29915)
#labels = getLabelsfromStocks(stockInfo, stockInfoDJI)
completeDF  = getcompleteDF(filelist)
completeDF.to_csv('DowJones_10K_012_basemethod.csv')
#print(completeDF)
#print(stockInfo)