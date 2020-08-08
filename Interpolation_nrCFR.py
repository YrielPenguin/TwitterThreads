import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tqdm
from collections import OrderedDict
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

root_path = '/'.join(sys.path[0].split('\\'))+'/'
data_path = root_path+'COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
data_path2 = root_path+'Documents/Science Projects/data_nrCFR/'
cols = ['Confirmed', 'Deaths', 'Recovered']

Example = 'France'
start = '05.01' ; date_start = start.split('.')[1]+'/'+start.split('.')[0]

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

def getMetrics(df, metrics = ['CFR', 'nrCFR_7'], n = 7):
    global z ; z = df
    for met in metrics:
        if '_' not in met:
            metrics[metrics.index(met)] = met + '_None'
    return [eval(met.split('_')[0]+'(z,n='+met.split('_')[1]+')').iloc[n:] for met in metrics]

class modelInterpolPoly():
    def __init__(self, degree=2):
        self.degree = degree
    def fit(self, x):
        self.pf = np.polyfit(range(len(x)),x,self.degree)
    def predict(self, x):
        return [sum([c*x**(self.degree-i) for i,c in enumerate(self.pf)]) for x in range(len(x))]
    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)
        
        
if __name__ == '__main__':
    if True:
        df = readFile(Example, date_start=date_start)
        y = getMetrics(df, metrics = ['nrCFR_7'], n = 7)[0].loc['01/05':'30/05']
        lr = LinearRegression()
        degs = [1,2,6,12] ; resps = ['A', 'B', 'C', 'D']
        
        for resp, deg in zip(resps, degs):
            X = np.reshape([[_**(d+1) for d in range(deg)] for _ in range(len(y.index))],
                             [-1,deg])
            
            n_splits = 10 ; error = 0
            kf = KFold(n_splits=n_splits, random_state = 42)
            kf.get_n_splits(X)
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                lr.fit(X_train, y_train)
                error += mse(lr.predict(X_test), y_test) / n_splits
            #A 0.0097 ; B 0.0011 ; C 0.015 ; D 14.78
            lr.fit(X, y.values)
            reslr = lr.predict(X)
            plt.plot(y.index, y, color = 'blue')
            plt.plot(y.index, reslr, color = 'green')
            plt.xticks(['01/05','30/05'])
            plt.ylabel('nrCFR 7days')
            plt.xlabel('Dates')
            plt.title('France : Réponse ' + resp)
            plt.figure()
            
    if True:
        df = readFile(Example, date_start=date_start)
        y = getMetrics(df, metrics = ['nrCFR_7'], n = 7)[0].loc['01/05':'30/05']
        lr = LinearRegression()
        degs = [1,12] ; resps = ['A', 'D']
        
        for resp, deg in zip(resps, degs):
            X = np.reshape([[_**(d+1) for d in range(deg)] for _ in range(len(y.index))],
                             [-1,deg])
            
            n_splits = 10 ; error = 0
            kf = KFold(n_splits=n_splits, random_state = 42)
            kf.get_n_splits(X)
            
            lr.fit(X, y.values)
            reslr = lr.predict(X)
            plt.plot(y.index, y, color = 'blue')
            plt.plot(y.index, reslr, color = 'green')
            plt.xticks(['01/05','30/05'])
            plt.ylabel('nrCFR 7days')
            plt.xlabel('Dates')
            
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                lr.fit(X_train, y_train)
                error += mse(lr.predict(X_test), y_test) / n_splits
                plt.plot(y.index[train_index], lr.predict(X_train), color='orange')
            plt.title('France : Réponse ' + resp + ' ; Error : ' + str(round(error,5)))
            plt.figure()
        
        
    if True:
        df = pd.DataFrame([[0.065,0.0187], [0.129,0.0751], [0.075,0.0714]], columns=['x','y'])
        lr = LinearRegression() ; y = df.y
        X = np.reshape(df.x.values,[-1,1])
        lr.fit(X, y)
        
        plt.plot(X, y, '*')
        plt.plot(X, lr.predict(np.reshape(df.x.values,[-1,1])), color='blue')
        n_splits = 3 ; error = 0
        kf = KFold(n_splits=n_splits, random_state = 42)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lr.fit(X_train, y_train)
            error += mse(lr.predict(X_test), y_test) / n_splits
            plt.plot(X_train, lr.predict(X_train), color='orange')
        plt.xlim([.9*min(X), 1.1*max(X)])
        plt.ylim([.9*min(y), 1.1*max(y)])
        lr.fit(X, y)
        plt.title('Linear Regression : R² ' + str(round(lr.score(X, y), 4))+ 
                  ' ; Quadratic Error : ' +str(round(error,4)))
        
        
    if True :
        dfs_by_countries = readFiles()
        lr = LinearRegression()
        degs = [1,2] ; resps = ['LR1', 'LR2']
        n_splits = 10
        kf = KFold(n_splits=n_splits, random_state = 42)
        
        for c in tqdm.tqdm(dfs_by_countries):
            dfs_by_countries[c] = getMetrics(dfs_by_countries[c], metrics = ['nrCFR_7'], n = 7)[0].loc['15/06': '29/06']
        
        errors = OrderedDict()
        for c in tqdm.tqdm(dfs_by_countries):
            y = dfs_by_countries[c]
            if sum(y!=y)!=0:
                print(c)
            else:
                errors[c]= {}
                for resp, deg in zip(resps, degs):
                    X = np.reshape([[_**(d+1) for d in range(deg)] for _ in range(len(y.index))],
                                     [-1,deg])
                    error = 0
                    kf.get_n_splits(X)
                    
                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        lr.fit(X_train, y_train)
                        error += mse(lr.predict(X_test), y_test) / n_splits
                        
                    errors[c][resp] = error
        #36 countries have nrCFR which contains missing values, assumption : becaus of there are no deaths
        #or recovered during one week few times, these countries are excluded for linear regression.
        
        df_errors = pd.DataFrame(None, columns = resps)
        for c in errors:
            df_errors.loc[c] = [errors[c][resp] for resp in resps]
            
            #LR1<LR2 : 57 ; LR2<LR1 : 68 ; == : 25
            #means : LR1 : 0.0034 ; LR2 : 0.0041
    
    
    
    if False:
        df = readFile(Example, date_start=date_start)
        model = LinearRegression()
        X = np.reshape([_ for _ in range(df.shape[0])],[-1,1])
        model.fit(X, df.Deaths.values)
        
        il = model.predict(X)
        plt.plot(df.index, df.Deaths, color = 'blue')
        plt.plot(df.index, il, color = 'red')
        plt.title('Cumulated Deaths by Covid-19 in France')
        plt.xlabel('Dates')
        plt.ylabel('Deaths')
        plt.xticks([df.index[0], df.index[-1]])
        plt.legend(['Data', 'Reg Linear 1'])
        
        model2 = LinearRegression() ; deg = 2
        X = np.reshape([[_**(d+1) for d in range(deg)] for _ in range(df.shape[0])],
                         [-1,deg])
        model2.fit(X, df.Deaths.values)
        
        il2 = model2.predict(X)
        plt.plot(df.index, df.Deaths, color = 'blue')
        plt.plot(df.index, il, color = 'red')
        plt.plot(df.index, il2, color = 'green')
        plt.title('Cumulated Deaths by Covid-19 in France')
        plt.xlabel('Dates')
        plt.ylabel('Deaths')
        plt.xticks([df.index[0], df.index[-1]])
        plt.legend(['Data', 'Reg Linear 1', 'Reg Linear 2'])
        
    
    
    
    
    
    
    
    
    