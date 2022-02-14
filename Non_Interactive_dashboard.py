#Importing Libaries

from math import comb
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import plotly.express as px
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output 
import dash_bootstrap_components as dbc


#Setting the stocks that will be used in the dashboard
stock1 = "JPM"
stock2 = "WTI"

#Downloading the relevant stock data for JP Morgan and WTI Crude
#data = yf.download([stock1, stock2])
#Previewing datasets
#print(data.head())

# #This revelead that JPM has stock data prior to WTI

#print(data.tail())
 
#There are no mising values towards the end of the dataset, therefore
#the Nan values from the past can be removed, as it will be pointless to 
#compare JPM stock data against nothing.

#Putting the stocks in a list to make it easier to call the stocks
stock_list = [stock1,stock2]

#Function to get stock data
def get_data(ticker):
    #Downloading the relevant stock data for JP Morgan and WTI Crude
    data = yf.download(ticker)
    #Dropping Na values
    data.dropna(inplace = True)
    #Only looking at close data.
    data_filtered = data['Adj Close']
    return data_filtered

#Function to create intial stock graphs 
def create_graphs(ticker):
    #For each ticker in the list of tickers
    for x in ticker:
        #As JPM is a stock and WTI is a Commodity they do not share a "Stock Price"
        if x == 'JPM':
            #Only use the JPM column of data
            filtered_JPM = data_filtered[x]

            #Creating line graph for JPM Stock Price
            fig_JPM = px.line(filtered_JPM, x = filtered_JPM.index , y = "JPM")
            #Adding a title that is centered in the middle
            fig_JPM.update_layout(
                        margin = dict(t = 50, b = 50, l = 25, r = 25))
            #Creating functional buttons and a range slider to allow interactivity with the graph.
            fig_JPM.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count = 5, label = "5y", step="year", stepmode="backward"),
                dict(step="all")
                    ])
                )
            )
            #Adding a title to the Y-Axis
            fig_JPM.update_yaxes(fixedrange = False, title_text = "Price ($USD)")

            #Calculating returns for JPM
            returns_JPM = np.log(filtered_JPM/filtered_JPM.shift(1))
            returns_JPM.dropna(inplace = True)

            #Creating returns graph
            fig_JPM_returns = px.line(returns_JPM, x = returns_JPM.index , y = "JPM")
            fig_JPM_returns.update_layout(
                        margin = dict(t = 50, b = 50, l = 25, r = 25))
            fig_JPM_returns.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                dict(count = 7, label = "1w", step = "day", stepmode = "backward"),
                dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
                dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
                dict(count = 2, label = "2y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5y", step = "year", stepmode ="backward"),
                dict(step ="all")
                    ])
                )
            )
            fig_JPM_returns.update_yaxes(fixedrange = False, title_text = " % Returns")

        else:
            #Filtering data for only WTI
            filtered_WTI = data_filtered[x]
            #Creating line graph for WTI Stock Price
            fig_WTI = px.line(filtered_WTI, x = filtered_WTI.index , y = "WTI")
            fig_WTI.update_layout(
                        margin = dict(t = 50, b = 50, l = 25, r = 25))
            fig_WTI.update_traces(line_color = 'red')
            fig_WTI.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count = 5, label = "5y", step="year", stepmode="backward"),
                dict(step="all")
                    ])
                )
            )
            fig_WTI.update_yaxes(fixedrange = False, title_text = "Price ($USD)")

            #Calculating returns for WTI
            returns_WTI = np.log(filtered_WTI/filtered_WTI.shift(1))
            returns_WTI.dropna(inplace = True)

            #Creating returns graph
            fig_WTI_returns = px.line(returns_WTI, x = returns_WTI.index , y = "WTI")
            fig_WTI_returns.update_layout(
                        margin = dict(t = 50, b = 50, l = 25, r = 25))
            fig_WTI_returns.update_traces(line_color = 'red')
            fig_WTI_returns.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                dict(count = 7, label = "1w", step = "day", stepmode = "backward"),
                dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
                dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
                dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
                dict(count = 2, label = "2y", step = "year", stepmode = "backward"),
                dict(count = 5, label = "5y", step = "year", stepmode ="backward"),
                dict(step ="all")
                    ])
                )
            )

            fig_WTI_returns.update_yaxes(fixedrange = False, title_text = " % Returns")
            
    return fig_JPM, returns_JPM, fig_JPM_returns, fig_WTI, returns_WTI, fig_WTI_returns

data_filtered = get_data(stock_list)

fig_JPM, returns_JPM, fig_JPM_returns, fig_WTI, returns_WTI, fig_WTI_returns = create_graphs(stock_list)

#Function for direct stock comparison graph
def stock_comparison(data):
    stockfig = px.line(data_filtered, x = data_filtered.index , y = ["JPM", "WTI"])
    stockfig.update_layout(
                        legend_title_text = "Tickers",
                        margin = dict(t = 50, b = 50, l = 25, r = 25))
    stockfig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 7, label = "1w", step = "day", stepmode = "backward"),
            dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
            dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
            dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
            dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
            dict(count = 2, label = "2y", step = "year", stepmode = "backward"),
            dict(count = 5, label = "5y", step = "year", stepmode ="backward"),
            dict(step ="all")
            ])
        )
    )
    stockfig.update_yaxes(fixedrange = False, title_text = " Price ($USD)")
    return stockfig

stockfig = stock_comparison(data_filtered)

#function to create a dataframe with both JPM and WTI returns
def combined_returns():
    #Calulating the Log returns between the day before and the next
    combined_returns_df = np.log(data_filtered/data_filtered.shift(1))
    combined_returns_df.dropna(inplace = True)
    return combined_returns_df

combined_returns_df = combined_returns()

#Creating Correlation scatter graph
def corr_scatter(data, stock1_returns, stock2_returns):
    stock_scatter = px.scatter(data, stock1_returns, stock2_returns, trendline= "ols")
    stock_scatter.update_layout(
                                margin = dict(t = 50, b = 50, l = 25, r = 25))
    stock_scatter.update_xaxes(title_text = "JPM Returns")
    stock_scatter.update_yaxes(title_text = "WTI Returns")
    return stock_scatter

stock_scatter = corr_scatter(data_filtered, returns_JPM, returns_WTI)

#Calculating correlation, rolling   6-month correlation
def corr_calculations(returns_df):
    #Calculating the overall correlation
    correlation = returns_df.corr()
    #Setting a 6-month window for the correlation calculation.
    rolling_corr = returns_df[stock1].rolling(window = 183).corr(returns_df[stock2])
    return correlation, rolling_corr

correlation, rolling_corr = corr_calculations(combined_returns_df)

#Creating rolling correlation graph

def rolling_corr_graph(roll_corr):
    rolling_corr_fig = px.line(roll_corr, x= roll_corr.index, y = 0)
    rolling_corr_fig.update_layout(
                        margin = dict(t = 50, b = 50, l = 25, r = 25))
    rolling_corr_fig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 7, label = "1w", step = "day", stepmode = "backward"),
            dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
            dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
            dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
            dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
            dict(count = 2, label = "2y", step = "year", stepmode = "backward"),
            dict(count = 5, label = "5y", step = "year", stepmode ="backward"),
            dict(step ="all")
                ])
            )
        )
    rolling_corr_fig.update_yaxes(fixedrange = False, title_text = " Correlation")
    return rolling_corr_fig

rolling_corr_fig = rolling_corr_graph(rolling_corr)

#fig_JPM.show()
#fig_JPM_returns.show()
#fig_WTI.show()
#fig_WTI_returns.show()
#stockfig.show()
#stock_scatter.show()
#rolling_corr_fig.show()

#Building web application using Dash

app = dash.Dash()


app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H2('Correlation Overview', className='text-center text-primary, mb-3'))),  # header row
        
        dbc.Row([  # start of 2nd row
            dbc.Col([  # 1st column on 2nd row
            html.H5('JPM Stock Price History', className='text-center'),
            dcc.Graph(id = 'JPM_stock',
                      figure = fig_JPM,
                      style={'height':350})

            ], width={'size': 6, 'offset': 0, 'order': 1}),  # width 1st column on 2nd row
            dbc.Col([  # 2nd column on 2nd row
            html.H5('WTI Crude Oil Price History', className='text-center'),
            dcc.Graph(id='WTI_Price',
                      figure=fig_WTI,
                      style={'height':350})
            ], width={'size': 6, 'offset': 0, 'order': 2}),  # width 2nd column on 2nd row
        ]),  # end of second row
        
        dbc.Row([  # start of 3rd row
            dbc.Col([  # 1st column on 3rd row
                html.H5('JPM Log Returns', className='text-center'),
                dcc.Graph(id='JPM_returns',
                      figure=fig_JPM_returns,
                      style={'height':350}),
            ], width={'size': 6, 'offset': 0, 'order': 1}),  # width 1st column on 3rd row
            dbc.Col([  # 2nd column on 3rd row
                html.H5('WTI Log Returns', className='text-center'),
                dcc.Graph(id='WTI_returns',
                      figure = fig_WTI_returns,
                      style={'height':350}),
            ], width={'size': 6, 'offset': 0, 'order': 2}),  # width 2nd column on 3rd row
        ]),# end of 3rd row
        dbc.Row([  # start of 4th row
            dbc.Col([  # 1st column on 4th row
                html.H5('JPM Vs WTI', className='text-center'),
                dcc.Graph(id='JPM_vs_WTI',
                      figure=stockfig,
                      style={'height':350}),
            ], width={'size': 6, 'offset': 0, 'order': 1}),  # width 1st column on 4th row
            dbc.Col([  # 2nd column on 4th row
                html.H5('Correlation', className='text-center'),
                dcc.Graph(id='Correlation_comaprison',
                      figure = stock_scatter,
                      style={'height':350}),
            ], width={'size': 6, 'offset': 0, 'order': 2}),  # width 2nd column on 4th row
        ]),# end of 4th row
        dbc.Row([  # start of 5th row
            dbc.Col([  # 1st column on 5th row
                html.H5('6-Month Rolling Correlation', className='text-center'),
                dcc.Graph(id='6-month_Corr',
                      figure=rolling_corr_fig,
                      style={'height':350}),
            ], width={'size': 12, 'offset': 0, 'order': 1}),  # width 1st column on 5th row
        ])    # end of 5th row  
        
    ],fluid=True)


if __name__ == '__main__':
    app.run_server()