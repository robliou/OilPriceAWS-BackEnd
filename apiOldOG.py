import time
from flask import Flask, jsonify
from flask import request
from flask import Response
from flask_cors import CORS
from datetime import date
import datetime
import json


from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web

import requests
import pandas as pd


from flask_wtf import FlaskForm

from wtforms import StringField, SubmitField

from wtforms.validators import DataRequired

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor

import seaborn as sns

import plotly

import plotly.express as px

import base64
from io import BytesIO


app = Flask(__name__, static_folder='api/build')
app.secret_key = "secKeyy"
CORS(app)


class RegForm(FlaskForm):
    tickerOne = StringField('tickerOne', validators=[DataRequired()])
    tickerTwo = StringField('tickerTwo')
    begDate = StringField('begDate',  validators=[DataRequired()])
    endDate = StringField('endDate', validators=[DataRequired()])


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}


@app.route('/api/residuals', methods=["POST"])
def find_the_score():
    """ reg_form = RegForm() """

    JSONdata = request.get_json()

    print("this is JSONdata", JSONdata)

    start = JSONdata['begDate']

    end = JSONdata['endDate']

    tickerOne = JSONdata['tickerOne']

    tickerTwo = JSONdata['tickerTwo']

    tickerThree = JSONdata['tickerThree']

    tickerFour = JSONdata['tickerFour']

    tickerFive = JSONdata['tickerFive']

    if tickerTwo == "":
        stocks = [tickerOne,  'NEO']

    elif tickerThree == "":
        stocks = [tickerOne, tickerTwo, 'NEO']

    elif tickerFour == "":
        stocks = [tickerOne, tickerTwo, tickerThree, 'NEO']

    elif tickerFive == "":
        stocks = [tickerOne, tickerTwo, tickerThree, tickerFour, 'NEO']

    else:
        stocks = [tickerOne, tickerTwo, tickerThree,
                  tickerFour, tickerFive, 'NEO']

    print("this is ticker one", tickerOne)

    df = web.DataReader(tickerOne, 'yahoo',
                        start, end)

    df = df.dropna()
    """ drops N/A """

    ls_key = 'Close'
    """ ls_key specifies the column name you want to select from the Excel dataset! """

    cleanData = df[ls_key]

    dataFrame = pd.DataFrame(cleanData)

    print("This is dataFrame", dataFrame)

    print("This is dataFrame column names", dataFrame.columns.values)

    # features = dataFrame.loc[:, stocks[:-1]] <---- X'd out this

    features = dataFrame

    """ stocks[:-1] selects all values from array except for last item """

    print('this is features:', features)

    X = features

    print('this is df.columns', df.columns)

    y = features.copy()

    # Create trend features
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        constant=True,  # the intercept
        order=1,        # quadratic trend
        drop=True,      # drop terms to avoid collinearity
    )
    X = dp.in_sample()  # features for the training data

    print('this is X', X)
    print('this is y', y)

    X_test = X.iloc[:, [0]]
    print('this is x_test', X_test)
    print('this is X column names', X.columns.values)
    y_train = y.loc[:]
    print('this is y_train', y_train)

    # settings
    plt.style.use('seaborn')
    plt.rcParams["figure.figsize"] = (16, 8)

    # Isolate deseasoning only
    seasonal_df = X.merge(y_train, how="right", on="Date")
    #seasonal_df.index = pd.to_datetime(seasonal_df.index)
    # Above makes date the index!

    print('this is seasonal_df', seasonal_df)
    print('this is seasonal_df column names', seasonal_df.columns.values)
    print('this is column names-y', seasonal_df.columns.values)
    print('this is sesasonal_df[0]', seasonal_df.index)

    # pandas datetimeindex docs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html
    # efficient way to extract year from string format date
    seasonal_df['month'] = pd.DatetimeIndex(seasonal_df.index).month
    print('this is seasonal_df-after adding month', seasonal_df)

    # calculate the seasonal component
    seasonal_df["seasonality"] = "NanN"
    print('this is seasonal_df with blank columns', seasonal_df)

    seasonal_df["seasonality"] = seasonal_df.groupby(
        "month")['Close'].transform("mean")
    # This calculates the seasonality, per https://towardsdatascience.com/time-series-diy-seasonal-decomposition-f0b469afed44

    seasonal_df["trendy"] = seasonal_df['Close'].rolling(window=13).mean()
    print("this is seasonal_trendy", seasonal_df["trendy"])
    seasonal_df.head(15)
    print('here is another seasonal_df', seasonal_df)

    decomposition = seasonal_decompose(seasonal_df['Close'],
                                       model='additive',
                                       period=(12))
    # 252 * 8.6 = 2167

    print('this is seasonality from decompose', decomposition.seasonal)
    print('this is seasonality from decompose[0]', decomposition.seasonal)
    decomposition.plot()

    seasonal_decomp = decomposition.seasonal
    seasonal_resid = decomposition.resid
    seasonal_trend = decomposition.trend

    df_season = pd.DataFrame(
        {'date': seasonal_decomp.index, 'values': seasonal_decomp.values})
    df_season['date'] = df_season['date'].astype(str)
    df_season = df_season.to_dict()
    print('this is seasonality from df_season', df_season)

    fig = px.line(df_season, x="date", y="values",
                  title='Seasonal Decomposition')

    graphJSON = plotly.io.to_json(fig, pretty=True)
    # https://plotly.github.io/plotly.py-docs/generated/plotly.io.to_json.html
    # This is what/all you need to properly export plots to JS!

    #df_season = df_season.to_json()
    df_resid = pd.DataFrame(
        {'date': seasonal_resid.index, 'values': seasonal_resid.values})
    df_resid = df_resid.to_json()
    df_trend = pd.DataFrame(
        {'date': seasonal_trend.index, 'values': seasonal_trend.values})
    df_trend = df_trend.to_json()

    json.dumps({'data': df_season})

    #decompPlot = decomposition.plot()
    #('this is decompPlot', decompPlot)

    #html = str(mpld3.fig_to_html(decompPlot))
    #
    #html = decompPlot.savefig('my_plot.png', format='png')

    #print('this is html', html)

    print('this is graphJSON', graphJSON)

    return graphJSON


if __name__ == '__main__':
    app.run()
