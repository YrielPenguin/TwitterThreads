import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tqdm
from collections import OrderedDict
from scipy import stats

root_path = '/'.join(sys.path[0].split('\\'))+'/'
data_path = root_path+'COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
data_path2 = root_path+'Documents/Science Projects/data_nrCFR/'
cols = ['Confirmed', 'Deaths', 'Recovered']

Preprocessing_by_date = False
Preprocessiong_by_country = True
Example = 'France'
start = '05.01' ; date_start = start.split('.')[1]+'/'+start.split('.')[0]

def CFR(df, n = None):
    return df['Deaths'] / (df['Confirmed'])

def nrCFR(df, n = 7):
    ndeaths = df['Deaths'] - df['Deaths'].shift(n)
    nhealed = df['Recovered'] - df['Recovered'].shift(n)
    return ndeaths / (ndeaths + nhealed)

def nCFR(df, n = 7):
    ndeaths = df['Deaths'] - df['Deaths'].shift(n)
    nconfirmed = df['Confirmed'] - df['Confirmed'].shift(n)
    return ndeaths / nconfirmed

def nrDeaths(df, n = 7):
    return df['Deaths'] - df['Deaths'].shift(n)

def nrHealed(df, n = 7):
    return df['Recovered'] - df['Recovered'].shift(n)

# =============================================================================
# def nmeanDeaths(df, n = 7):
#     return (df['Deaths'] - df['Deaths'].shift(n)) / n
# 
# def nmeanHealed(df, n = 7):
#     return (df['Recovered'] - df['Recovered'].shift(n)) / n
# =============================================================================

def getMetrics(df, metrics = ['CFR', 'nrCFR_7'], n = 7):
    global z ; z = df
    for met in metrics:
        if '_' not in met:
            metrics[metrics.index(met)] = met + '_None'
    return [eval(met.split('_')[0]+'(z,n='+met.split('_')[1]+')').iloc[n:] for met in metrics]

def plotMetrics(df, metrics = ['CFR', 'nrCFR_7'], n = 7, lancetgate = True, rlancetgate = False,
                title = '', __nrDeaths = True, __nrHealed = True, retractimpactlg = True):
    colors = ['blue', 'green', 'red', 'black']
    legends = [_ if '_None' not in _ else _.split('_')[0] for _ in metrics]
    ress = getMetrics(df, metrics = metrics, n = n)
    ind = df.index[n:]
    xticks = [ind[0], ind[-1]]
    if __nrDeaths or __nrHealed:
        fig = plt.figure(figsize = (15, 8))
        fig.add_subplot(223)
    else:
        plt.figure(figsize = (20, 10))
    for _, res in enumerate(ress):
        plt.plot(ind, res, color=colors[_])
    if lancetgate:
        plt.vlines(['22/05', '04/06'], ymin=0,
                   ymax=max([max(res) for res in ress]), color = 'grey')
        legends += ['LancetGate']
        xticks += ['22/05', '04/06']
    if rlancetgate:
        plt.vlines(['09/06', '22/06'], ymin=0,
                   ymax=max([max(res) for res in ress]), color = 'purple')
        legends += ['LancetGate+18j']
        xticks += ['09/06', '22/06']
    if rlancetgate:
        plt.vlines(['15/06', '29/06'], ymin=0,
                   ymax=max([max(res) for res in ress]), color = 'yellow')
        legends += ['Retract+18j+/-7j']
        xticks += ['15/06', '29/06']
    plt.legend(legends)
    plt.xlabel('Dates')
    plt.ylabel('Metrics')
    plt.xticks(xticks, rotation=30)
    if (title != '') & (not __nrDeaths) & (not __nrHealed):
        plt.title(title)
        
    if __nrDeaths or __nrHealed:
        host = fig.add_subplot(221)
        legends= [] ; nrd = [0] ; nrh = [0]
        if __nrDeaths & __nrHealed:
            nrd = nrDeaths(df, n).iloc[n:] ; nrh = nrHealed(df, n).iloc[n:]
            plt.plot(ind, nrd, color='black') ; plt.ylabel('Deaths')
            ax2=host.twinx() ; plt.plot(0,0, color='black')
            ax2.plot(ind, nrh, color='orange') ; ax2.set_ylabel('Healed')
            legends += ['Deaths since '+str(n)+' days', 'Healed since '+str(n)+' days']
        elif __nrDeaths:
            nrd = nrDeaths(df, n).iloc[n:]
            plt.plot(ind, nrd, color='black') ; plt.ylabel('Deaths')
            legends += ['Deaths since '+str(n)+' days']
        elif __nrHealed:
            nrh = nrHealed(df, n).iloc[n:]
            plt.plot(ind, nrh, color='orange') ; plt.ylabel('Healed')
            legends += ['Healed since '+str(n)+' days']
        if lancetgate:
            plt.vlines(['22/05', '04/06'], ymin=0,
                       ymax=max(nrd+nrh), color = 'grey')
        if rlancetgate:
            plt.vlines(['09/06', '22/06'], ymin=0,
                       ymax=max(nrd+nrh), color = 'purple')
        plt.legend(legends)
        plt.title(title)
        plt.xticks('')
        
def lancetGateKSTest(df, metrics = ['CFR', 'nrCFR_7'],
        periods = [['28/05', '08/06'],['09/06', '22/06'],['23/06', '06/07']], alpha = 0.01):
    ress = getMetrics(df, metrics = metrics, n = n) ; pvals = []
    if (len(periods) == 3) & (None not in periods):
        blg = df.iloc[:,0].loc[periods[0][0]:periods[0][1]].index
        lg = df.iloc[:,0].loc[periods[1][0]:periods[1][1]].index
        alg = df.iloc[:,0].loc[periods[2][0]:periods[2][1]].index
        for r in ress:
            pvals += [stats.ks_2samp(r.loc[blg], r.loc[lg]).pvalue,
                 stats.ks_2samp(r.loc[alg], r.loc[lg]).pvalue]
    elif (len(periods) == 2) or (None in periods):
        blg = df.iloc[:,0].loc[periods[0][0]:periods[0][1]].index
        lg = df.iloc[:,0].loc[periods[1][0]:periods[1][1]].index
        for r in ress:
            pvals += [stats.ks_2samp(r.loc[blg], r.loc[lg]).pvalue]
    return pvals

def lancetGateKSTestDFS(dfs_by_countries, metrics = ['CFR', 'nrCFR_7'],
        periods = [['28/05', '08/06'],['09/06', '22/06'],['23/06', '06/07']], alpha = 0.01):
    types = ['blg_', 'alg_']
    if None == periods[0]:
        types = ['alg_']
    elif None == periods[-1]:
        types = ['blg_']
    pvalues = pd.DataFrame(np.NaN, index = dfs_by_countries.keys(),
                           columns = [j+i for i in metrics for j in types[:(len(periods)-1)]])
    for country in tqdm.tqdm(dfs_by_countries):
        pvalues.loc[country] = lancetGateKSTest(dfs_by_countries[country],
                   metrics=metrics, periods=periods, alpha=alpha)
    return pvalues

def readFile(Example, date_start = date_start, cut_before_start = True):
    df = pd.read_csv(data_path2+Example+'.csv',sep=';')
    df.index = df.iloc[:,0] ; df.drop(df.columns[0], axis = 1, inplace = True)
    if date_start in df.index:
        if df.isna().sum().sum()!=0:
            print('There are missing values here !')
        if cut_before_start:
            df = df.loc[date_start:]
        return df
    else:
        print('Starting Date is not included !')
        return None
    
def readFiles(date_start = date_start, nan_killer = True, cut_before_start = True):
    files = os.listdir(data_path2)
    dfs_by_countries = OrderedDict()
    for f in files:
        df = pd.read_csv(data_path2+f,sep=';')
        df.index = df.iloc[:,0] ; df.drop(df.columns[0], axis = 1, inplace = True)
        if date_start in df.index:
            if cut_before_start:
                df = df.loc[date_start:]
            if not nan_killer:
                dfs_by_countries[f.split('.')[0]] = df
            elif df.isna().sum().sum()==0:
                dfs_by_countries[f.split('.')[0]] = df
            else:
                 print(f)
    if ('Congo (Brazzaville)' in dfs_by_countries) & ('Congo (Kinshasa)' in dfs_by_countries):
        dfs_by_countries['Congo'] = dfs_by_countries['Congo (Kinshasa)'] + dfs_by_countries['Congo (Brazzaville)']
        del dfs_by_countries['Congo (Kinshasa)'] ; del dfs_by_countries['Congo (Brazzaville)']
    return dfs_by_countries

def countriesLowerLGFromKSTest(dfs_by_countries, lgkstdfs,
                    periods = [['28/05', '08/06'],['09/06', '22/06'],['23/06', '06/07']]):
    lower_dif_blg = []
    if periods[0] != None:
        dif_blg = list(lgkstdfs.loc[lgkstdfs['blg_nrCFR_7']<0.01].index)
        for c in dif_blg:
            ress = getMetrics(dfs_by_countries[c], metrics=['nrCFR_7'])[0]
            ress = pd.DataFrame(ress, index=dfs_by_countries[c].index)
            ress_blg = ress.loc[periods[0][0]:periods[0][1]]
            ress_lg = ress.loc[periods[1][0]:periods[1][1]]
            if (ress_blg.sum() > ress_lg.sum()).iloc[0]:
                lower_dif_blg += [c]
            
    lower_dif_alg = []
    if periods[2] != None:
        dif_alg = list(lgkstdfs.loc[lgkstdfs['alg_nrCFR_7']<0.01].index)
        for c in dif_alg:
            ress = getMetrics(dfs_by_countries[c], metrics=['nrCFR_7'])[0]
            ress = pd.DataFrame(ress, index=dfs_by_countries[c].index)
            ress_alg = ress.loc[periods[2][0]:periods[2][1]]
            ress_lg = ress.loc[periods[1][0]:periods[1][1]]
            if (ress_alg.sum() > ress_lg.sum()).iloc[0]:
                lower_dif_alg += [c]
            
    return lower_dif_blg, lower_dif_alg

if __name__ == '__main__':
    
    if Preprocessiong_by_country :
        files = os.listdir(data_path)
        for f in files:
            if 'csv' not in f:
                del files[files.index(f)]
        
        dfs = OrderedDict() ; countries = set()
        
        for f in tqdm.tqdm(files):
            if float(start) <= (int(f.split('-')[0])+int(f.split('-')[1])/100):
                df = pd.read_csv(data_path+f)
                if 'Country_Region' in df.columns:
                    df['Country/Region'] = df['Country_Region']
                    df.drop('Country_Region', axis=1, inplace=True)
                df = df[cols + ['Country/Region']].groupby('Country/Region').sum(axis=1)
                countries = set(list(countries) + list(df.index))
            
            for country in countries:
                if country not in dfs:
                    dfs[country] = pd.DataFrame([], index = [], columns = cols)
                if country in df.index:
                    dfs[country].loc[f.split('.')[0]] = df.loc[country]
                else:
                    dfs[country].loc[f.split('.')[0]] = [np.NaN, np.NaN, np.NaN]
            
        for country in countries:
            dfs[country].index = [_.split('-')[1] + '/' + _.split('-')[0] for _ in dfs[country].index]
            dfs[country].to_csv(data_path2+country.split('*')[0]+'.csv',
                   sep=';')
             
    if Example != None:
        df = readFile(Example)
        #df.index = df.iloc[:,0] ; df.drop(df.columns[0], axis = 1, inplace = True)
        n = 7 ; metrics = ['CFR', 'nrCFR_7']
        plotMetrics(df, metrics=metrics, n=n, lancetgate=True, rlancetgate=False, retractimpactlg=True,
                    title=Example+' ; n = '+str(n), __nrDeaths=True, __nrHealed=True)
        
    if False:
        dfs_by_countries = readFiles()
        lgkstdfs = lancetGateKSTestDFS(dfs_by_countries, metrics = ['CFR', 'nrCFR_7'],
                    #periods = [['28/05', '08/06'],['09/06', '22/06'],['23/06', '06/07']], alpha = 0.01)
                    periods = [['31/05', '14/06'],['15/06', '29/06'],['30/06', '14/07']], alpha = 0.01)
        lgkstdfs['blg_nrCFR_7'].hist(bins=100) ; plt.title('Pvalues BLG')
        lgkstdfs['alg_nrCFR_7'].hist(bins=100) ; plt.title('Pvalues ALG')
        print((lgkstdfs['blg_nrCFR_7']<0.01).sum())
        print((lgkstdfs['alg_nrCFR_7']<0.01).sum())
        
        l_blg, l_alg = countriesLowerLGFromKSTest(dfs_by_countries, lgkstdfs)

                
        
    if False: #[['22/05', '04/06'] ,['09/06', '22/06']]
        #periods1 = [['29/04', '11/05'],['12/05', '25/05'],['26/05', '08/06']]
       # periods2 = [['23/06', '05/07'],['06/07', '19/07'],None]
        periods1 = [['16/04','30/04'],['01/05', '15/05'],['16/05', '30/05']]
        dfs_by_countries = readFiles()
        lgkstdfs1 = lancetGateKSTestDFS(dfs_by_countries, metrics = ['nrCFR_7'],
                    periods = periods1, alpha = 0.01)
        #lgkstdfs2 = lancetGateKSTestDFS(dfs_by_countries, metrics = ['nrCFR_7'],
                    #periods = periods2, alpha = 0.01)
        
        lgkstdfs1['blg_nrCFR_7'].hist(bins=100)
        lgkstdfs1['alg_nrCFR_7'].hist(bins=100)
        #lgkstdfs2['blg_nrCFR_7'].hist(bins=100)
        print((lgkstdfs1['blg_nrCFR_7']<0.01).sum())
        print((lgkstdfs1['alg_nrCFR_7']<0.01).sum())
        #print((lgkstdfs2['blg_nrCFR_7']<0.01).sum())
        
# =============================================================================
#         stats.ks_2samp(lgkstdfsB['blg_nrCFR_7'], lgkstdfs['blg_nrCFR_7']).pvalue
#         stats.ks_2samp(lgkstdfsA['blg_nrCFR_7'], lgkstdfs['blg_nrCFR_7']).pvalue
# =============================================================================
        
        l_b1, l_a1 = countriesLowerLGFromKSTest(dfs_by_countries, lgkstdfs1, periods1)
        #l_b2, l_a2 = countriesLowerLGFromKSTest(dfs_by_countries, lgkstdfs2, periods2)
    
    
    
    
    