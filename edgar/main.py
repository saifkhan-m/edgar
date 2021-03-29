from sec_edgar_downloader import Downloader
import os
import extractsection as es
import pandas as pd
def getAll10k(company):
    dl = Downloader()
    # Get all 10-K filings for Microsoft
    dl.get("10-K", company, amount=10)

def getsectionsCompany(path, company):
    lst1 = []
    for root, dirs, files in os.walk(os.path.join(path,company)):
        for file in files:
            if file.endswith(".txt"):
                lst1.append(os.path.join(root, file))
    df = pd.DataFrame(columns=['Company','Year','Business', 'Riskfactors', 'Legal', 'Management', 'Quantitative'])
    for file in lst1:
        with open(file) as f:
            year = '20'+file.split('10-K')[1].split('-')[1]
            data = f.read()
            data = es.clean_data(data)
            business, riskfactors, legal, management, quantitative = es.extractsections(data)
            df = df.append({'Company':company,'Year':year,'Business':business, 'Riskfactors':riskfactors,
                            'Legal':legal, 'Management':management, 'Quantitative':quantitative},
                           ignore_index=True)
    return df


def main():
    dow = ["MSFT"]
    df = pd.DataFrame(columns=['Company', 'Year', 'Business', 'Riskfactors', 'Legal', 'Management', 'Quantitative'])
    for company in dow:
        print('extracting datset')
        getAll10k(company)
        1/0
        print('processing and cleaning')
        retdf = getsectionsCompany('sec-edgar-filings', company)
        df = pd.concat([df, retdf], ignore_index=True)
        print('complete')
    df.to_csv('Apple.csv',index=False)


main()
#sentence = "edgar/sec-edgar-filings/MSFT/10-K/0001193125-11-200680"
#print(sentence.split("10-K"))