# Import all needed libraries
import pandas as pd # to manipulate dataframes
import numpy as np # for computation
import requests # to pull requests from the API
import json # for requests interpretation
import plotly # to plot graphs in the user interface

from flask import Flask # to create graphs objects
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import datetime # to manipulate datetimes
import predicator # to train models

app = Flask(__name__)

def load_data(tickers, today):
    '''
    Description :
    Load data the user want to see from the Yahoo finance web site with the Yahoo API. All the data is appended in a dataframe.

    INPUT :
        - tickers : list of desired tickers to request with the API

    OUTPUT : 
        - df : dataframe with all data of the desired tickers
    '''
    
    #parameters to pull requests
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"

    headers = {
    'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
    'x-rapidapi-key': "6f2bf57cd9mshdc2132958a6b924p107011jsnca8d5e653398"
    }
    period1 = pd.to_datetime(datetime.date(year=today.year - 1, month=today.month, day= today.day)).value
    period2 = pd.to_datetime(today).value
    df = pd.DataFrame(data = None)
    for ticker in tickers:
        query = {"frequency":"1d","filter":"history","period1":period1,"period2":period2,"symbol":ticker}
        response = requests.request('GET', url, headers=headers, params=query)
        df_response = pd.DataFrame(response.json()['prices'])
        try:
            df_response = df_response[['adjclose', 'close', 'date', 'high', 'low', 'open', 'volume']]       
        except:
            pass
        df_response['Company'] = ticker
        df = df.append(df_response, ignore_index = True, sort=False)
    df = df.rename({'adjclose':'Adj Close', 'close':'Close', 'date':'Date', 'high':'High', 'low':'Low', 'open':'Open', 'volume':'Volume'}, axis='columns')
    df['Data'] = 'Original'
    df['Date'] = pd.to_datetime(df.Date, unit='s').dt.date
    df = df[df.Date >= datetime.date(year=today.year - 1, month=today.month, day= today.day)]

    return df

def fill_df(df):
    '''
    Description :
    Some 'Adj Close' values are missing, I fill them with the mean of the previous and the next 'Adj Close' values for the visualization.

    INPUT :
        - df : the loaded dataframe from the API
    
    OUTPUT :
        - df : filled dataframe
    '''
    index_to_fill = df[df['Adj Close'].isnull() == True].index
    for index in index_to_fill:
        df['Adj Close'].loc[index] = np.mean([df['Adj Close'].loc[index - 1], df['Adj Close'].loc[index + 1]])
    
    return df


# load model
train_duration = 0
forecast = 0

def transform_to_date(string, vect, today):
    '''
    Description :
    Convert string to datetime

    INPUT : 
        - string : period to convert
        - vect : period for training or prediction
        - today : today's date
    
    OUTPUT :
        - date : date converted to datetime type
    '''
    mod = 1
    if vect == 'train': mod = -1
    if string == '1day':
        date = datetime.date(year=today.year, month=today.month, day= today.day +(1*mod))
    elif string == '1week':
        date = datetime.date(year=today.year, month=today.month, day= today.day +(7 * mod))
    elif string == '1month':
        date = datetime.date(year=today.year, month=today.month +(1 * mod), day= today.day)
    elif string == '3months':
        date = datetime.date(year=today.year, month=today.month +(3 * mod), day= today.day)
    elif string == '1year':
        date = datetime.date(year=today.year +(1 * mod), month=today.month, day= today.day)
    return date

def predict_new(df, train_duration, predicted_date, ticker, today):
    '''
    Description :
        - For a selected ticker, bags of knn and LR models are trained on the training duration selected
    to predict the values of the 'Adj Close' on the predicted due date. The mean of the model predictions 
    is computed to improve the global output.
        - Then the predicted value is appended in a new dataframe with the predicted due date.
        - The worst value (lowest and highest) are computed too with the prediction score obtained

    INPUT : 
        - df : the dataframe containing the raw market values
        - train_duration : the training duration selected by the user
        - predicted_date : the predicted date selected by the user
        - ticker : the ticker selected by the user
        - today : today's date

    OUTPUT :
        df2 = the new dataframe with the predicted value for the predicted date only
        df2_low = the new dataframe with the lowest predicted value for the predicted date only
        df2_high = the new dataframe with the highest predicted value for the predicted date only
    '''
    # convert train value
    start_train = transform_to_date(train_duration, 'train', today) - datetime.timedelta(predicted_date)
    
    # train model
    bags, score = predicator.main(df, today, start_train, predicted_date)

    #define the period to predict
    X_to_pred = df.drop(columns = ['Adj Close', 'Date', 'Data'])[: predicted_date]
    X_to_pred = predicator.prediction_preparation(X_to_pred)

    #predict the new value
    values_pred = []
    for bag in bags:
        values_pred.append(np.round(bags[bag].predict(X_to_pred), 2))
    df_pred = pd.DataFrame(values_pred)
    new_pred = df_pred.mean(axis=0)[-1:]

    #Create a new dataframe with the predicted values
    date_pred = [today + datetime.timedelta(days = i + 1) for i in range(predicted_date)]
    df2 = pd.DataFrame({'Date':date_pred[-1:]})
    df2['Adj Close'] = new_pred.iloc[0]
    df2['Company'] = ticker
    df2['Data'] = 'Prediction'
    
    # Compute the lowest and the highest values
    df2_low = low_high(df2, score, 'low')
    df2_high = low_high(df2, score, 'high')

    return df2, df2_low, df2_high

def low_high(df, score, vect):
    '''
    Description :
    Compute the lowest and highest value possible according to the value predicted and the score

    INPUT :
        - df : datarame of the value predicted
        - score : score obtained at the prediction
        - vect : 'low' or 'high'

    OUTPUT :
        - df : dataframe of the lowest or highest value predicted
    '''
    df3 = df.copy()
    if vect == 'low':
        df3['Adj Close'] = df['Adj Close'] * np.abs(score)
    else:
        df3['Adj Close'] = df['Adj Close'] / np.abs(score)
    return df3.copy()

def growth_fn(df, growth_period, today):
    '''
    Description : 
    Compute the growth rate for the growth period selected by the user

    INPUT :
        - df : dataframe of the data for the seleccted tickers
        - growth_period : period to compute the growth
        - today : today's date
    
    OUTPUT :
        - df : dataframe of the tickers sorted by the growth rate
    '''
    companies = df.Company.unique()
    growth_period = (today - transform_to_date(growth_period, 'train', today)).days
    growths = {}
    for company in companies:
        df2 = df[df.Company == company]['Adj Close']
        growth = df2[-1:].iloc[0] - df2[-growth_period-1:].iloc[0]
        growths[company] = growth
    df3 = pd.DataFrame(growths.items(), columns=['Company', 'Growth'])
    return df3.sort_values(by='Growth', ascending=False)

def generate_graph_data(df, df2, df2_low, df2_high, ticker):
    '''
    Description :
    Generate graphs for the visualizations from the data and the predicted data

    INPUT :
        - df : dataframe of the original data from the API
        - df2 : dataframe of the predicted data
        - df2_low : dataframe of the lowest predicted data
        - df2_high : dataframe of the highest predicted data
        - ticker : ticker selected by the client
    
    OUTPUT :
        - data : list of scatter plots to visualize

    '''
    dates = df[df.Company == ticker]['Date']
    values = df[df.Company == ticker]['Adj Close']
    dates_pred = df2[ticker]['Date']
    values_pred = df2[ticker]['Adj Close']
    values_low = df2_low[ticker]['Adj Close']
    values_high = df2_high[ticker]['Adj Close']

    data = [
        Scatter(
            x=dates,
            y=values,
            name=ticker
        ),
        Scatter(
            x=dates_pred,
            y=values_pred,
            name='Predicted value for ' + str(ticker)
        ),
        Scatter(
            x=dates_pred,
            y=values_high,
            name='Higher predicted values for ' + str(ticker)
        ),
        Scatter(
            x=dates_pred,
            y=values_low,
            name='Lower predicted values for ' + str(ticker)
        )
    ]
    
    return data
    
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    today = datetime.date.today()

    # Ticker selected by user
    tickers = request.args.get('tag', '').split(',')
    # default ticker
    #if tickers[0] == '': tickers = ['AAPL', 'AMZN', 'FB', 'NFLX', 'TSLA']
    if tickers[0] == '': tickers = ['AAPL']

    # Load data
    df = load_data(tickers, today)    

    # fill empty 'Adj Close' values for the visualization with the mean between the previous and the next values
    df = fill_df(df)

    # training and forecasting parameters selected by users
    train_duration = request.args.get('train', '')
    forecast = request.args.get('forecast', '')

    # prediction for each ticker selected
    df3 = {}
    df3_low = {}
    df3_high = {}

    # for each ticker
    for ticker in tickers:
        # create empty df
        df2 = pd.DataFrame(columns=['Date', 'Adj Close', 'Company', 'Data'])
        df2_low = pd.DataFrame()
        df2_high = pd.DataFrame()

        # if parameters have been selected, make predictions
        if train_duration != '' and train_duration !='0' and forecast != '' and forecast != '0':
            # convert prediction date from string to integer
            predicted_date = (transform_to_date(forecast, 'pred', today) - today).days
            #save the most recent value for visual purpose
            last_real = df[df.Company == ticker][['Date', 'Adj Close', 'Company', 'Data']][:1]

            # predict values with a growing shift until the predicted date for better precision for the first predictions
            for i in range(predicted_date):
                # make prediction for the predicted date, lowest prediction and highest prediction
                df_pred, df_pred_low, df_pred_high = predict_new(df[df.Company == ticker], train_duration, i + 1, ticker, today)
                # append prediction, lowest prediction and highest prediction to the previous predictions
                df2 = pd.concat([df2, df_pred])
                df2_low = pd.concat([df2_low, df_pred_low])
                df2_high = pd.concat([df2_high, df_pred_high])

            # append the most recent value to the predictions
            df2 = pd.concat([last_real, df2], axis=0, ignore_index=True)
            df2_low = pd.concat([last_real, df2_low], axis=0)
            df2_high = pd.concat([last_real, df2_high], axis=0)

        # if no parameters have been selected, leave the df empty
        else:
            df2 = df.drop(df.index)
            df2_low = df.drop(df.index)
            df2_high = df.drop(df.index)
        
        # for each ticker, append the predictions in a df
        df3[ticker] = df2
        df3_low[ticker] = df2_low 
        df3_high[ticker] = df2_high

    growth_period = request.args.get('growth_period', '')
    if growth_period =='': growth_period='1day'
    df_growth = growth_fn(df, growth_period, today)

    # extract data needed for visuals
    companies = df_growth.Company
    growth = df_growth.Growth
    
    # create visuals
    graph_data = [generate_graph_data(df, df3, df3_low, df3_high, ticker) for ticker in tickers]

    all_data = []
    for i, data in enumerate(graph_data):
        for scatter in data:
            all_data.append(scatter)

    # define the graphs parameters for visualizations
    graphs = [
        {
            'data': all_data,

            'layout': {
                'title': 'Value of the ' + str(tickers) + ' actions',
                'yaxis': {
                    'title': "Value of the stock option"
                },
                'xaxis': {
                    'title': "Date"
                }
            }
        },
    
        {
            'data': [
                Bar(
                    x=companies,
                    y=growth
                )
            ],

            'layout': {
                'title': 'Companies global growth',
                'yaxis': {
                    'title': "Companies"
                },
                'xaxis': {
                    'title': "Growth"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()