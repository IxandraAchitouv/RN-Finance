
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy import stats
import math
from numpy import linalg as LA
pd.core.common.is_list_like = pd.api.types.is_list_like
#import pandas_datareader as web
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from random import sample
plt.style.use('fivethirtyeight')
#import yfinance as yf
from yahoo_fin.stock_info import get_data
from scipy.stats import kurtosis
from scipy.stats import skew
import yfinance as yf

import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
import graphviz
import re
import pydot #https://github.com/pydot/pydot
from scipy.stats import kurtosis
from scipy.stats import skew

def read_downloaddata(download,start,end):

    if download==1:
        # for the stocks 
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        df1 = yf.download(tickers.Symbol.str.strip().to_list(),start,end, auto_adjust=True)['Close']

       
        # Merge the result with df3 on 'Date'
        df =df1 #pd.merge(df_temp, df3, on=['Date'])

       
        df.fillna(method='ffill', inplace=True) # fill nan with the last observed value
        df.dropna(axis = 1, how = 'any',inplace=True)
        
        df.to_csv('data_stocks/df_stocks_'+start+'_'+end+'.csv', index=True)
        
        # for the dictionary per sector 
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        dft = pd.read_html(str(table))[0]
       
        df_filtered500 = dft[['Symbol', 'GICS Sector']]
        df_filtered500['GICS Sector']=df_filtered500['GICS Sector'].apply(lambda x: x+' S&P 500')


        url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        # Send a GET request to the URL
     #   response = requests.get(url)
        # Parse the HTML content of the page with BeautifulSoup
    #    soup = BeautifulSoup(response.text, 'html.parser')
        # Find the table with the class 'wikitable sortable'
    #    table = soup.find('table', {'class': 'wikitable sortable'})
        # Use pandas to read the HTML table
    #    dft = pd.read_html(str(table))[0]
        # Select only the columns 'Symbol' and 'GICS Sector'
   #     df_filtered600 = dft[['Symbol', 'GICS Sector']]
   #     df_filtered600['GICS Sector']=df_filtered600['GICS Sector'].apply(lambda x: x+' S&P 600')
        
        # URL of the Wikipedia page
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        # Send a GET request to the URL
        response = requests.get(url)
        # Parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the table with the class 'wikitable sortable'
        table = soup.find('table', {'class': 'wikitable sortable'})
        # Use pandas to read the HTML table
        dft = pd.read_html(str(table))[0]
        # Select only the columns 'Symbol' and 'GICS Sector'
        df_filtered400 = dft[['Symbol', 'GICS Sector']]
        df_filtered400['GICS Sector']=df_filtered400['GICS Sector'].apply(lambda x: x+' S&P 400')

        
        df_filtered=df_filtered500 #pd.concat([df_filtered500, df_filtered400], ignore_index=True)
        values=df_filtered['GICS Sector'].tolist()
        keys=df_filtered['Symbol'].tolist()
        dic_sector=dict(map(lambda i,j : (i,j) , keys,values))
        # Save the DataFrame to a CSV file
        df_filtered.to_csv('../data_stocks/S&Pcompanies_dic.csv', index=False)
        
        return df, dic_sector
        
    else:
        df_filtered=pd.read_csv('../data_stocks/S&Pcompanies_dic.csv')
        values=df_filtered['GICS Sector'].tolist()
        keys=df_filtered['Symbol'].tolist()
        dic_sector=dict(map(lambda i,j : (i,j) , keys,values))

        df=pd.read_csv('../data_stocks/df_stocks_'+start+'_'+end+'.csv', index_col=[0],
    parse_dates=[0])

        return df,dic_sector


def log_return(Stocki):
    Ri=[]
    T=len(Stocki)
    for i in range(1,T):
        Ri.append(np.log(Stocki[i])-np.log(Stocki[i-1]))
   # Ri=np.log(list_stock_prices).diff() 
    return Ri

def computeCorrM(Stocks):
    #si=Stocks[:,stocki]
    Returns=[]
    Ret=[]
    Nstocks=len(Stocks[0])
    Time=len(Stocks[:,0])
    for i in range(Nstocks):
        si=Stocks[:,i]
        lr=[]
        for t in range(1,Time):
            lr.append(np.log(si[t])-np.log(si[t-1]))

        Ret.append(lr)
        #ri=(lr-np.mean(lr))/np.std(lr)
        ri=lr
        Returns.append(ri)
    
    
    CorrM=np.corrcoef(Returns) #np.cov(Returns)
    return CorrM,Ret
    
def MarchenkoPastur(Q,Lam,sig_r):
    #sig_r=1 # as the correlation matrix diagonal/variance is one 
  
    print('sig_r=',sig_r)
    Lmin=sig_r**2*(1-1/np.sqrt(Q))**2
    Lmax=sig_r**2*(1+1/np.sqrt(Q))**2
    Pdf=[]
    for x in Lam:
        if(Lmax-x)<0 or (x-Lmin)<0:
            Pdf.append(0)
        else:
            Pdf.append(Q/2./sig_r**2/np.pi*np.sqrt((Lmax-x)*(x-Lmin))/x)

    pdfnor=[i/sum(Pdf) for i in Pdf]
    return Lmin,Lmax,Pdf

def simulate_stock_priceGBM(S0, mu, sigma, T):
    """
    Simulate a stock price using Geometric Brownian Motion (GBM).

    Parameters:
    S0 (float): Initial stock price.
    mu (float): Drift coefficient (expected return).
    sigma (float): Volatility (standard deviation of returns).
    T (float): Total time period.
    dt (float): Time step.

    Returns:
    np.array: Simulated stock prices.
    """
    np.random.seed(seed=None)
    dt=1.#0.001
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)  # Time grid
    W = np.random.normal(size=N)  # Generate random numbers for Brownian motion
    W = np.cumsum(W)*np.sqrt(dt)  # Cumulative sum to generate Wiener process

    X = (mu-0.5*sigma**2)*t+sigma*W  # Log returns
    S = S0*np.exp(X)  # Stock prices

    return S
    
def simulate_stock_priceGBM_corr(S0, mu, sigma, T):
    """
    Simulate a stock price using Geometric Brownian Motion (GBM).

    Parameters:
    S0 (float): Initial stock price.
    mu (float): Drift coefficient (expected return).
    sigma (float): Volatility (standard deviation of returns).
    T (float): Total time period.
    dt (float): Time step.

    Returns:
    np.array: Simulated stock prices.
    """
    seedi=4523
    coef=1
    dt=1.#0.001
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)  # Time grid
    
    np.random.seed(seed=None)
    W1 = np.random.normal(size=N)
    np.random.seed(seedi)
    W2 = np.random.normal(size=N)
    

    W=[]
    
    for i in range(N):
        W.append((W1[i]*(1.-coef)+coef*W2[i]))
    
    
    W = np.cumsum(W)*np.sqrt(dt)  # Cumulative sum to generate Wiener process

    X = (mu-0.5*sigma**2)*t+sigma*W  # Log returns
    S = S0*np.exp(X)  # Stock prices

    return S


def findTc(lambda_mu,a0):
    dt = 0.01  # Time step
    time_steps = 1000  # Number of time steps
    H_t=[]
    N = len(lambda_mu) 
    for it in range(time_steps):
        t=dt*it
        ee = np.exp(-2. * np.array(lambda_mu) * t)
        ss=1./N*sum(ee)
        H_t.append(ss)

    # Compute Laplace Transform at s=0
    s=0.0
    H_t = np.array(H_t)
    t_vals = np.arange(time_steps) * dt
    Laplace_H_t = np.sum(H_t * np.exp(-s * t_vals)) * dt

    print(Laplace_H_t)
    #a0=100
    Tc=0.5*a0*1/Laplace_H_t
    return Tc



    
def Stocks_beta(Beta,Stocks):
    CorrM,Ret=computeCorrM(Stocks)
    StocksRD=[]
    Time=len(Stocks[:,0])
    Nstocks=len(Stocks[0])
    if Beta>=0:
        for istock in range(Nstocks):
            s1=simulate_stock_priceGBM(Stocks[0,istock], np.mean(Ret[istock]), np.std(Ret[istock]), Time)
            StocksRD.append(s1)
        StocksRD=np.array(StocksRD).T
    else:
        for istock in range(Nstocks):
            s1=simulate_stock_priceGBM_corr(Stocks[0,istock], np.mean(Ret[istock]), np.std(Ret[istock]), Time)
            StocksRD.append(s1)
        StocksRD=np.array(StocksRD).T
        

    
    Stockstest=[]    
    for istock in range(Nstocks):
        if Beta>=0:
            s1=Beta*StocksRD[:,istock]+(1-Beta)*Stocks[:,istock]
        else:
            s1=-Beta*StocksRD[:,istock]+(1+Beta)*Stocks[:,istock]
        Stockstest.append(s1)
    Stockstest=np.array(Stockstest).T

    return Stockstest
    
def findTc_beta(a0,Stocks):
 
    beta_var=[-0.99,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    Tc_var=[]
    for Beta in beta_var:
        print('*'*10)
        Stockstest=Stocks_beta(Beta,Stocks)
        
        CorrMtest,Rettest=computeCorrM(Stockstest)
        eigenvalues, z = LA.eig(CorrMtest)
        eigenvalues.sort()
        #print(eigenvalues)
        # find Lcut:
        counts, bin_edges = np.histogram(eigenvalues, bins=[min(eigenvalues)+i*0.05 for i in range(50)])
        for i in range(len(counts)):
            if counts[i]<1:
                Lcut=bin_edges[i]
                break
            
        # compute Lambda_mu
        lambda_mu=[]
        for l in eigenvalues:
            if l<Lcut:
                linv=1./l-1./Lcut
                lambda_mu.append(linv)
            else:
                break
                
        # compute Tc
        Tc=findTc(lambda_mu,a0)
        print('Beta=',Beta,'nb of lambda=',len(lambda_mu),'Lcut=',Lcut,'Tc=',Tc,'a0=',a0)
        Tc_var.append(Tc)

    return Tc_var,beta_var
    
    
def solveq_mu(lambda_mu,T,a0):
    # Parameters
    N = len(lambda_mu)  # Number of variables
    #T =Temperature
    dt = 0.0005  # Time step
    time_steps = 1000  # Number of time steps
    
    # Initialization to 1 (or -1)
    q_mu =np.array([1*i/i for i in range(1,N+1)])  #np.random.rand(N)
    q_mu_history = []

    
    # Euler-Maruyama method
    for t in range(time_steps):
        # Compute l(t) with h_0=-1 and h_1=1 h_{i>1}=0
   
        l_t = -1+(1./a0*1/N) * np.sum(q_mu**2)
        
        
        # Generate eta_mu for each time step
        eta_mu = np.sqrt(2 * T/dt) * np.random.randn(N)
        
        # Update q_mu using Euler-Maruyama
        #q_mu += dt * (- (lambda_mu + l_t) * q_mu) + dt * eta_mu
        if np.sum(q_mu**2)<1.e+10:
            q_mu += dt * (- (lambda_mu + l_t) * q_mu) + dt * eta_mu
            

        # Store history for plotting
        q_mu_history.append(q_mu.copy())
       
    # Convert history to numpy array for easier manipulation
    q_mu_history = np.array(q_mu_history)
    
    return q_mu_history,np.arange(time_steps) * dt

def make_qmu_averaged_realisation(Beta,a0,ToverTc,Stocks,binss):
    Stocks_b=Stocks_beta(Beta,Stocks)
    CorrMtest,Rettest=computeCorrM(Stocks_b) # compute the correlation matrix from the stock return
    eigenvalues, z = LA.eig(CorrMtest)     # compute the eigenvalues
    eigenvalues.sort()
    
    #compute Lcut
    counts, bin_edges = np.histogram(eigenvalues, bins=[min(eigenvalues)+i*binss for i in range(150)])
    Lcut = bin_edges[np.argmax(counts < 2)] # compute the largest eigenvalue of the continious spectrum we put 1 as a more restrictive threshold then 0
    
    print('Lcut=',Lcut)
    lambda_mutemp=[]
    
    for l in eigenvalues:
        if l<Lcut:
            linv=1./l-1./Lcut
            lambda_mutemp.append(linv)
    Nbulk=len(lambda_mutemp)
    print('Nbulk=',Nbulk)
    
    lambda_mu=[]
    for l in eigenvalues:
        if eigenvalues[0]-l<1./np.sqrt(Nbulk) and l<Lcut:
            linv=1./l-1./Lcut
            lambda_mu.append(linv)

    if len(lambda_mu)<0.05*Nbulk:
        lambda_mu=lambda_mutemp
        
    
    print('nb of lambda=',len(lambda_mu),'Lcut=',Lcut,'nb lambdaall=',len(eigenvalues))
    print('Lmax=',lambda_mu[0])
    N=len(lambda_mu)
   
    
    #plt.figure(figsize=(10, 6))
    #plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", alpha=0.7)
    #plt.savefig('hist_bet='+str(Beta)+'_ToverTc='+str(ToverTc)+'.png')
    #plt.axvline(x=Lcut, color='r', linestyle='--', linewidth=2, label='Lcut')
    Tc=findTc(lambda_mu,a0)
    print('Tc=',Tc)
    
    T=Tc*ToverTc
    q_mu_history,tt=solveq_mu(lambda_mu,T,a0)
    #print(len(q_mu_history),len(q_mu_history[0]))
    #print(q_mu_history)
    Tall=len(tt)
    qmumean_ieta=np.zeros([Tall,N])
    Neta=1000

    T=Tc*ToverTc
    for ieta in range(Neta):
        q_mu_history,tt=solveq_mu(lambda_mu,T,a0)
        qmumean_ieta+=q_mu_history
    qmumean_ieta=qmumean_ieta/Neta

    
    # Example time and indices
    time = np.arange(qmumean_ieta.shape[0])
    indices = np.arange(qmumean_ieta.shape[1])
    
    # Convert to a DataFrame
    dfmu = pd.DataFrame(qmumean_ieta, columns=indices, index=time)
  
    # Optionally, save the DataFrame to a CSV file
    dfmu.to_csv("dataoutput/qumuNeta="+str(Neta)+"ToTc="+str(ToverTc)+"beta="+str(Beta)+"a0="+str(a0)+".csv",index=False)
    return dfmu


def read_qmu_averaged_realisation(Beta,a0,ToverTc,Stocks):
    dfmu=pd.read_csv("dataoutput/qumuNeta=1000ToTc="+str(ToverTc)+"beta="+str(Beta)+"a0="+str(a0)+".csv")
    return dfmu
