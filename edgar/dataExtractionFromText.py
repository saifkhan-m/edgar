import os
from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
yf.pdr_override()
import numpy as np

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

def getLabelsfromStocks(stockInfo):
    closingList = stockInfo['Close'].tolist()
    reslist = [0]
    for i in range(len(closingList)-1):
        reslist.append(closingList[i+1]-closingList[i])

    stockInfo['label'] = np.array(reslist)
    stockInfo['label'] = stockInfo["label"].apply(lambda x: 0 if x < 0 else (2 if x > 0 else 1))
    stockInfo.reset_index(level=0, inplace=True)
    stockInfo = stockInfo.rename(columns={'Date': 'year'})
    return stockInfo


def getcompleteDF(lst1):
    cols = ['Name','CIK','year', 'Ticker','Business','Risk Factors','Legal Proceedings','MDA_FCRO','QQD_MR', 'label']
    companyInfoDF = getCompanyInfo()
    listofcompanyDF = []
    for index, row in companyInfoDF.iterrows():
        stockInfo = getStockInfo(row['Ticker'])
        stockInfowithlabels = getLabelsfromStocks(stockInfo)
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

def getStockInfo(company,start="2008-01-01",end="2020-12-31"):
    info = pdr.get_data_yahoo(company, start=start, end=end, interval="1mo")
    crit2 = info.index.map(lambda x: x.month == 1)
    info = info[crit2]
    info.index = pd.to_datetime(info.index, format='%Y-%m-%d').year
    info = info.dropna()
    info['Ticker'] = company
    df = info[['Ticker','Close']]
    return df

filelist = getfilelist('../SEC-EDGAR-text/output_files_examples/batch_0004')
#companyInfoDF = getCompanyInfo()
#stockInfo = getStockInfo('DOW')
#compandf = getInfofromfiles(filelist, 29915)
#labels = getLabelsfromStocks(stockInfo)
completeDF  = getcompleteDF(filelist)
completeDF.to_csv('DowJones_10K_012.csv')
#print(completeDF)
#print(stockInfo)