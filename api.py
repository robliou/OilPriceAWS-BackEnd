from flask import Flask, jsonify
from flask import request
from flask import Response
from flask_cors import CORS
from datetime import date
import datetime
import json
import requests

from statsmodels.tsa.seasonal import seasonal_decompose

import plotly

import plotly.express as px

import plotly.graph_objs as go
import pandas as pd

from functools import reduce

from dbnomics import fetch_series, fetch_series_by_api_link

import yfinance as yf

import json
import urllib


app = Flask(__name__, static_folder='api/build')
app.secret_key = "secKeyy"
CORS(app)


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


@app.route('/')
def index():
    return app.send_static_file('index.html')



@app.route('/api/currentPrice', methods=["GET"])
def getOilPrice():
    oilPrice = yf.Ticker("CL=F")
    currentPrice = jsonify(oilPrice.info)
    print('this is current price', currentPrice)

    return currentPrice


@app.route('/api/historicalPrice', methods=["GET"])
def getHistoricalPrice():
    df3 = fetch_series([
        'EIA/PET/F000000__3.M'
    ], max_nb_series=180).query("period >= '2015-01-01'")

    df3['period'] = df3['period'].astype(int)

    df3['value'] = df3['value'].astype(int)

    fig = go.Figure(px.line(df3,  x="period", y="value",
                            title="Historical oil prices"))

    # fig.add_trace(go.Line(x=merged_df.original_period,
    #                      y=merged_df.value, name='EIA', marker=dict(color='rgba(171, 50, 96, 0.6)')))

    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="top",
                                  y=1.02,
                                  xanchor="left",
                                  x=1))

    historicalPrices = plotly.io.to_json(fig, pretty=True)

    return historicalPrices


@app.route('/api/priceForecasts', methods=['GET'])
def getPriceForecasts():
    df_forecast = pd.read_json("dataPrediction.json")

    df_forecast['period'] = df_forecast['year'].astype(int)

    df_forecast['EIA'] = df_forecast['EIA'].astype(int)

    df_forecast['OPEC'] = df_forecast['OPEC'].astype(int)

    df_forecast['OECD'] = df_forecast['OECD'].astype(int)

    df_forecast['IEA'] = df_forecast['IEA'].astype(int)

    print('this is df_forecast', df_forecast)

    fig = go.Figure(px.line(df_forecast,  x="period", y="value",
                            title="Price forecasts"))

    # fig.add_trace(go.Line(x=merged_df.original_period,
    #                      y=merged_df.value, name='EIA', marker=dict(color='rgba(171, 50, 96, 0.6)')))

    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="top",
                                  y=1.02,
                                  xanchor="left",
                                  x=1))

    priceForecasts = plotly.io.to_json(fig, pretty=True)

    return priceForecasts


@app.route('/api/newSupply', methods=["GET"])
def getNewSupply():
    df7 = pd.read_csv('./NewSupply2022.csv')

    df7['year'] = df7['year'].astype(int)

    df7['value'] = df7['value'].astype(int)

    print('this is df6', df7)

    figSupply = go.Figure(px.line(df7,  x="year", y="value", color="region",
                                  title="New Supply Wedge"))

    # fig.add_trace(go.Line(x=merged_df.original_period,
    #                      y=merged_df.value, name='EIA', marker=dict(color='rgba(171, 50, 96, 0.6)')))

    figSupply.update_layout(legend=dict(orientation="h",
                                        yanchor="top",
                                        y=1.02,
                                        xanchor="left",
                                        x=1))

    newSupply = plotly.io.to_json(figSupply, pretty=True)

    return newSupply


@app.route('/api/four')
def getChartFour():

    df8 = fetch_series([
        'EIA/IEO.2021/LOWOILPRICE.SUP_LIQ_TOT_MBPD_TOT_MBPD.A',
        'EIA/IEO.2021/HIGHOILPRICE.SUP_LIQ_TOT_MBPD_TOT_MBPD.A'

    ], max_nb_series=70)
    print('this is df8', df8)
    print('this is df8.columns()',  list(df8.columns))

    df8['original_period'] = df8['original_period'].astype(int)
    df8 = df8.drop('period', axis=1)
    df8 = df8.drop('original_value', axis=1)
    df8.rename(columns={'value': 'alt_value'})

    df6 = pd.read_csv('./CrudePredictions.csv')

    df6['original_period'] = df6['original_period'].astype(int)

    df6['alt_value'] = df6['alt_value'].astype(int)


    merged_df = df8.merge(df6, how="right", on="original_period")


    merged_df.tail()


    fig = go.Figure(px.line(df6,  x="original_period", y="alt_value", color="agency_forecast",
                            title="hello"))


    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1))

    figFive = plotly.io.to_json(fig, pretty=True)

    return figFive


@app.route('/api/merged', methods=["GET"])
def mergedPrices():

    df3 = fetch_series([
        'EIA/PET/F000000__3.M'
    ], max_nb_series=180)

    # chart for crude oil price monthly - seasonality extraction

    # 10 sin/cos pairs for "A"nnual seasonality

    fig = go.Figure(px.line(df3,  x="period", y="value",
                            title="hello"))

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list()
            )
        ]
    )

    decomposition = seasonal_decompose(df3['value'],
                                       model='additive',
                                       period=(12))

    decomposition.plot()

    seasonal_observed = decomposition.observed
    seasonal_decomp = decomposition.seasonal
    seasonal_resid = decomposition.resid
    seasonal_trend = decomposition.trend

    df_season = pd.DataFrame(
        {'date': seasonal_decomp.index, 'values': seasonal_decomp.values})
    df_season['date'] = df_season['date'].astype(str)
    print('this is seasonality from df_season', df_season)

    df_season.rename(columns={"values": "values_season"}, inplace=True)

    graphJSON = plotly.io.to_json(fig, pretty=True)
    # https://plotly.github.io/plotly.py-docs/generated/plotly.io.to_json.html
    # This is what/all you need to properly export plots to JS!

    df_resid = pd.DataFrame(
        {'date': seasonal_resid.index, 'values': seasonal_resid.values})
    df_resid['date'] = df_resid['date'].astype(str)

    df_resid.rename(columns={"values": "values_resid"}, inplace=True)

    df_resid = df_resid.sort_values('date')

    df_trend = pd.DataFrame(
        {'date': seasonal_trend.index, 'values': seasonal_trend.values})
    df_trend['date'] = df_trend['date'].astype(str)

    df_trend.rename(columns={"values": "values_trend"}, inplace=True)

    df_observed = pd.DataFrame(
        {'date': seasonal_observed.index, 'values': seasonal_observed.values})
    df_observed['date'] = df_observed['date'].astype(str)

    df_observed.rename(columns={"values": "values_observed"}, inplace=True)

    data_frames = [df_observed, df_trend, df_season, df_resid]


    df_merged = df_observed.merge(df_trend, how="left").merge(
        df_season, how="left").merge(df_resid, how="left")


    df_merge = df_merged.to_json()

    return df_merge


@app.route('/api/consumption', methods=["GET"])
def consumption():

    df3 = fetch_series([
        'EIA/INTL/53-1-WORL-TBPD.M'
    ], max_nb_series=180)

    fig = go.Figure(px.line(df3,  x="period", y="value",
                            title="hello"))

    decomposition = seasonal_decompose(df3['value'],
                                       model='additive',
                                       period=(12))
    # 252 * 8.6 = 2167

    decomposition.plot()

    seasonal_observed = decomposition.observed
    seasonal_decomp = decomposition.seasonal
    seasonal_resid = decomposition.resid
    seasonal_trend = decomposition.trend

    df_season = pd.DataFrame(
        {'date': seasonal_decomp.index, 'values': seasonal_decomp.values})
    df_season['date'] = df_season['date'].astype(str)

    df_season.rename(columns={"values": "values_season"}, inplace=True)

    graphJSON = plotly.io.to_json(fig, pretty=True)
    # https://plotly.github.io/plotly.py-docs/generated/plotly.io.to_json.html
    # This is what/all you need to properly export plots to JS!

    df_resid = pd.DataFrame(
        {'date': seasonal_resid.index, 'values': seasonal_resid.values})
    df_resid['date'] = df_resid['date'].astype(str)

    df_resid.rename(columns={"values": "values_resid"}, inplace=True)

    df_trend = pd.DataFrame(
        {'date': seasonal_trend.index, 'values': seasonal_trend.values})
    df_trend['date'] = df_trend['date'].astype(str)

    df_trend.rename(columns={"values": "values_trend"}, inplace=True)

    df_observed = pd.DataFrame(
        {'date': seasonal_observed.index, 'values': seasonal_observed.values})
    df_observed['date'] = df_observed['date'].astype(str)

    df_observed.rename(columns={"values": "values_observed"}, inplace=True)

    data_frames = [df_observed, df_trend, df_season, df_resid]

    date_dummy = df_observed.loc[:, 'date']
    # creates dummy dates in df_observed
    extracted_col = df_observed['date']
    # extracts 'date' column from df_observed

    df_dates = df3.loc[:, 'original_period']
    # get all dates, since seasonal_decomposition deletes dates from dataset! Then add back in to merged_df

    df_dates = pd.DataFrame(df_dates)

    # turns df_dates from a series to a dataframe- necessary step

    df_merged = df_observed.merge(df_trend, how="left").merge(
        df_season, how="left").merge(df_resid, how="left").join(df_dates, how="left")

    df_merged = df_merged.merge(df_dates, left_index=True, right_index=True)

    # need to add df_dates separately to df_merged because it doesn't have a common index with others..?

    df_merged = df_merged.drop('date', axis=1)
    # dropping the date prevented duplicated columns; erasing the dual line problem

    df_merged = df_merged.to_json()

    return df_merged


@app.route('/api/production', methods=["GET"])
def production():
    df_production = fetch_series([
        'EIA/INTL/53-1-WORL-TBPD.A'


    ], max_nb_series=70)
    df_production.head()

    df_production = df_production.dropna()

    df_production.reset_index(drop=True, inplace=True)

    decomposition = seasonal_decompose(df_production['value'],
                                       model='additive',
                                       period=(12))

    decomposition.plot()

    seasonal_observed = decomposition.observed
    seasonal_decomp = decomposition.seasonal
    seasonal_resid = decomposition.resid
    seasonal_trend = decomposition.trend

    df_season = pd.DataFrame(
        {'date': seasonal_decomp.index, 'values': seasonal_decomp.values})
    df_season['date'] = df_season['date'].astype(str)

    df_season.rename(columns={"values": "values_season"}, inplace=True)

    df_season = df_season.dropna()

    df_resid = pd.DataFrame(
        {'date': seasonal_resid.index, 'values': seasonal_resid.values})
    df_resid['date'] = df_resid['date'].astype(str)

    df_resid.rename(columns={"values": "values_resid"}, inplace=True)

    df_resid = df_resid.dropna()

    df_trend = pd.DataFrame(
        {'date': seasonal_trend.index, 'values': seasonal_trend.values})
    df_trend['date'] = df_trend['date'].astype(str)

    df_trend.rename(columns={"values": "values_trend"}, inplace=True)

    df_trend = df_trend.dropna()

    df_observed = pd.DataFrame(
        {'date': seasonal_observed.index, 'values': seasonal_observed.values})
    df_observed['date'] = df_observed['date'].astype(str)

    df_observed.rename(columns={"values": "values_observed"}, inplace=True)

    df_observed = df_observed.dropna()

    data_frames = [df_observed, df_trend, df_season, df_resid]

    date_dummy = df_observed.loc[:, 'date']
    # creates dummy dates in df_observed
    extracted_col = df_observed['date']
    # extracts 'date' column from df_observed

    df_dates = df_production.loc[:, 'original_period']
    # get all dates, since seasonal_decomposition deletes dates from dataset! Then add back in to merged_df

    df_dates = pd.DataFrame(df_dates)

    df_dates = df_dates.dropna()

    # turns df_dates from a series to a dataframe- necessary step

    df_merged_production = df_observed.merge(df_trend, how="left").merge(
        df_season, how="left").merge(df_resid, how="left").join(df_dates, how="left")

    df_merged_production = df_merged_production.merge(
        df_dates, left_index=True, right_index=True)

    # need to add df_dates separately to df_merged because it doesn't have a common index with others..?

    df_merged_production = df_merged_production.drop('date', axis=1)
    # dropping the date prevented duplicated columns; erasing the dual line problem

    df_merged_production = df_merged_production.fillna(0)

    df_merged_production = df_merged_production.to_json()

    return df_merged_production
