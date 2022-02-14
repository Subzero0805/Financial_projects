"""
Adaptation of Modern Portfolio theory
to produce a the "Efficient Frontier" to optimise Portfolios.
"""

#Import relevant libaries
from os import linesep
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import scipy.optimize as sc
import pandas as pd
import plotly.graph_objects as go

#Import data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start = start, end = end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


stocks = ['SPY','AAPL','BKCH','ICLN','AQWA','VTI','VBMFX']
#Blank list of stocks
# stocks = []
# # try block to handle the exception
# try:
#     stocks = []
#     choice = input("Would you like to add a stock? Y/N \n")
#     while choice == 'Y':
#         stocks.append(input("Please enter a Ticker Symbol:" ))
#         choice = input("Would you like to add another stock? Y/N \n")
#     print("Here is your selected stock list: {lst} ".format(lst = stocks.strip()))
# # if N is chosen then the stock list is printed to the user
# except:
#     print("Here is your selected stock list: {lst} ".format(lst = stocks))


#Function to calculate Portfolio Performance - weights = % weighting of each asset
def portfolioPerformance(weights, meanReturns, covMatrix):
    #to calculate the returns, sumating the mean returns with the weights, over total trading days.
    returns = np.sum(meanReturns * weights)*252
    #to calculate the standard deviation - we need to use Portfolio Variance - w^TΣw
    #np.dot() allows for Matrix multiplication, weights.T transposes the weights matrix/list
    # np.dot(covMatrix,weights) calculated first and then it's multiplied against weights.T
    #np.sqrt(252) is required because the volitility * sqrt(T) rule, for annualised returns
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(252)
    return returns, std

#Calculating the negative Sharpe ratio
def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns,covMatrix)
    return -(pReturns - riskFreeRate)/pStd


#Calculate maximum sharpe ratio, by minimising the negative SR by changing the weights.
#Using Scipy, following resource for minimisation, using SLSQP:
#Minimise -https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
#Sequntial Least SQuares Programming (SLSQP) Algorithm -
#https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    #ensuring the sum of all weights = 1 by ensuring that np.sum(x) - 1 = 0
    constraints = ({'type': 'eq', 'fun': lambda x : np.sum(x) - 1})
    bound = constraintSet
    #For each asset, we are ensuring to use these bounds, this can allow for
    #all assets to have a weighting.
    bounds = tuple(bound for asset in range(numAssets))
    #minimise the negative SR, parameter of how to minimise providing an inital guess of what the
    #weights should be, weights total num of Assets * 1/numAssets allows to calcualte equal allocation
    #of assets
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args = args,
                            method='SLSQP', bounds = bounds, constraints = constraints)
    return result

#Calculating Portfolio Variance, i.e. std
def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

#Minimise the portolio variance by adjusting the weights of the assets
#Method is the same as Sharpe Ratio just for Portfolio Variance
def minimiseVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    #Risk Free Rate is not definied in Portfolio performance hence not used below.
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x : np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args = args,
                            method='SLSQP', bounds = bounds, constraints = constraints)
    return result


#Defining a stocklist, so far all stocks must have the same currency and formatted the same.
#stocks = ['SPY','AAPL','BKCH','ICLN','AQWA','VTI','VBMFX']

#Defining the end date, set to today (current day)
endDate = dt.datetime.now()
#Defining the start date, intital value used it 1 year prior to today.
num = int(input("How many days prior to today would you like to consider? (1 year = 365)\n"))
startDate = endDate - dt.timedelta(days=num)

# defining the weights matrix, how to weight each asset in the portfolio
# need to turn the list into an array in order to be used as a matrix
# weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])
# returns, std = portfolioPerformance(weights, meanReturns,covMatrix)
# print(round(returns*100,2), round(std*100),2)
# result = maxSR(meanReturns, covMatrix)
# maxSR, maxWeights = result['fun'],result['x']
# print("Max SR: {maxSR} Maximum Weights: {maxWeights}".format(maxSR = maxSR, maxWeights = maxWeights))
# minVarResult = minimiseVariance(meanReturns, covMatrix)
# minVAR, minVarWeights = minVarResult['fun'],minVarResult['x']
# print(minVAR, minVarWeights)

meanReturns, covMatrix = getData(stocks,start = startDate, end=endDate)
#Return portolio return


#Calculates the optimal asset weighting for each return target with minimal variance
def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet = (0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    #this function sits here as we are defining the meanReturns and covMatrix within the overall function
    def portfolioReturn(weights):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]

    #Only interested in the section that is above the minimum volility for a given portfolio return
    constraints = ({'type': 'eq', 'fun': lambda x : portfolioReturn(x) - returnTarget},
                    {'type': 'eq', 'fun': lambda x : np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    optResult = sc.minimize(portfolioVariance, numAssets * [1/numAssets],args = args,
                            method = 'SLSQP', bounds = bounds, constraints = constraints)
    return optResult



#Return the max sharpe ratio, minimum volitility and the efficient frontier
def calculatedResults(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0,1)):
    #Calculate the returns based on the MaxSR ratio weightings
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_Returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns,covMatrix)    
    #Tabulate the allocations per symbol and allocation using Pandas
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'],index=meanReturns.index,columns = ['weightings'])
    #accessing the weightings column to covert to % and round to whole numbers.
    maxSR_allocation.weightings = [round(i*100,0) for i in maxSR_allocation.weightings]
    

    #Calculate for the minimum Variance Portolio
    minVol_Portfolio = minimiseVariance(meanReturns, covMatrix)
    minVol_Returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns,covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'],index=meanReturns.index,columns = ['weightings'])
    minVol_allocation.weightings = [round(i*100,0) for i in minVol_allocation.weightings]
    

    #creating a list portfolios between min variance and maxSR
    ListOfTargetReturns = []
    targetReturns = np.linspace(minVol_Returns,maxSR_Returns,20)
    #iterating through that list to calculate the weightings for each target return
    for target in targetReturns:
        #returning fun, as above code optimises for variance.
        ListOfTargetReturns.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
    
    maxSR_Returns, maxSR_std = round(maxSR_Returns*100,2), round(maxSR_std*100,2)

    print("""The Maximum Sharpe Ratio possible is: {SR}
with a annualised Volitility of: {std}
and an annualised return of {Return}.
In order to achieve these results, you should weight your portfolio like so:
             {allocation}\n \n""".format(SR = round((maxSR_Portfolio['fun']*-1),4), std = maxSR_std, 
             Return = maxSR_Returns, allocation = maxSR_allocation))

    minVol_Returns, minVol_std = round(minVol_Returns*100,2), round(minVol_std*100,2)

    print("""The minimum possible annualised volitility is : {std}
with a Sharpe Ratio of {SR}
and with an annualised return of {Return}.
In order to achieve these results, you should weight your portfolio like so:
             {allocation}""".format(std = minVol_std, SR = round(minVol_Portfolio['fun'],4), 
             Return = minVol_Returns, allocation = minVol_allocation))
    return maxSR_Returns, maxSR_std, maxSR_allocation , minVol_Returns, minVol_std, minVol_allocation, ListOfTargetReturns, targetReturns

#Return a graph that plots the minVol, maxSR and efficient frontier
def EE_graph(meanReturns, covMatrix, riskFreeReturn = 0, constraintSet = (0,1)):
    maxSR_Returns, maxSR_std, maxSR_allocation , minVol_Returns, minVol_std, minVol_allocation, ListOfTargetReturns, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0,1))
    
    #Max SR
    MaxSharpeRatio = go.Scatter(
        name = "Maximum Sharpe Ratio",
        mode = "markers",
        x = [maxSR_std],
        y = [maxSR_Returns],
        marker = dict(color= "red",size=14,line=dict(width=3,color= "black"))
    )

    #min Vol
    MinVol = go.Scatter(
        name = "Minimum Volitility",
        mode = "markers",
        x = [minVol_std],
        y = [minVol_Returns],
        marker = dict(color= "green",size=14,line=dict(width=3,color= "black"))
    )

    #efficient Frontier
    EF_Curve = go.Scatter(
        name = "Efficient Frontier",
        mode = "lines",
        x = [round(ef_std*100,2) for ef_std in ListOfTargetReturns],
        y = [round(target*100,2) for target in targetReturns],
        line = dict(color= "black", width=4, dash='dashdot')
    )

    data = [MaxSharpeRatio, MinVol, EF_Curve]
    
    layout = go.Layout(
        title = "Portfolio Optimisation",
        yaxis = dict(title = "Annualised Return (%)"),
        xaxis = dict(title = "Annualised Volitility (%)"),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder = "normal",
            bgcolor = "#E2E2E2",
            bordercolor = "black",
            borderwidth = 2
        ),
        width = 800 ,
        height = 600)
    fig = go.Figure(data=data, layout = layout)
    return fig.show()

#print(calculatedResults(meanReturns, covMatrix))



EE_graph(meanReturns, covMatrix)
