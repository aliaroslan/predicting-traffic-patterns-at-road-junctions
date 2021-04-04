import streamlit as st
import pandas as pd
import numpy as np
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly import tools
import plotly.offline as py
import plotly.express as px

#arima/sarima
from pathlib import Path
import sys
import statsmodels as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from datetime import datetime
import calendar
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

#lstm/cnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import math
from torchsummaryX import summary


siteHeader = st.beta_container()
dataSet = st.beta_container()
newFeatures = st.beta_container()
modelTraining_arima = st.beta_container()
modelTraining_sarima = st.beta_container()
modelTraining_lstm = st.beta_container()
modelTraining_cnn = st.beta_container()
modelTraining_hybrid = st.beta_container()
echo = st.echo()

st.markdown(
    """
    <style>
    .main {
    background-color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

background_color = '#000000'

@st.cache
def get_data(filename):
    get_data = pd.read_csv(filename)
    return get_data

with siteHeader:
    st.image('../../capstone project/streamlit ui/traffic-header.jpg')
    st.title("""Predicting **Traffic Patterns** at **Road Junctions**""")
    st.markdown('In this project, we are predicting traffic patterns in each of these four junctions for the next 4 months.')

with dataSet:
    st.header('Dataset ')
    # st.text('This dataset was taken from Kaggle: https://www.kaggle.com/vetrirah/ml-iot?select=train_ML_IOT.csv')
    st.markdown('This dataset was taken from: ' +
    """<a href="https://www.kaggle.com/vetrirah/ml-iot?select=train_ML_IOT.csv">Kaggle.com</a>
    """,
    unsafe_allow_html=True,)

    st.markdown("Data Description:")
    desc_table = pd.DataFrame({
        'Variable': ['ID', 'DateTime', 'Junction', 'Vehicles'],
        'Description': ['Unique ID', 'Hourly Datetime Variable', 'Junction Type', 'Number of Vehicles (Target)']
    })
    st.table(desc_table)


    st.markdown("Below shows a snippet of the dataset:")

    train_data = get_data('../../capstone project/kaggle dataset/train_ML_IOT.csv')
    traffic_data = pd.DataFrame(train_data.head(10))
    # st.write(train_data.head())

    # test for go.plotly to display table
    fig_dataset = go.Figure(data=go.Table(
        header=dict(values=list(traffic_data[['DateTime', 'Junction', 'Vehicles', 'ID']].columns),
        fill_color='#242526', align='center'),
        cells=dict(values=[traffic_data.DateTime, traffic_data.Junction, traffic_data.Vehicles, traffic_data.ID],
        fill_color='#3A3B3C', align='center')
    ))

    fig_dataset.update_layout(autosize=True, height=260, margin=dict(l=5, r=5, b=10, t=10), paper_bgcolor=background_color)

    st.write(fig_dataset)
    
    with modelTraining_lstm:
    st.header('Model training: LSTM')
    # st.text('LSTM model was used...')

    split_ratio = 0.05
    num_epochs = 1
    window_size = 5
    batch_size = 5
    n_step = 2
    learning_rate = 0.001

    traffic_junction1 = train_data.loc[train_data["Junction"] == 1]
    traffic_junction2 = train_data.loc[train_data["Junction"] == 2]
    traffic_junction3 = train_data.loc[train_data["Junction"] == 3]
    traffic_junction4 = train_data.loc[train_data["Junction"] == 4]

    new_traffic = train_data.set_index(pd.to_datetime(train_data["DateTime"]))

    new_traffic = new_traffic.drop(columns = ['DateTime','ID'])
    st.markdown('Dataset with columns DateTime, Junction and Vehicles:')
    # table currently doesn't include DateTime
    st.write(new_traffic.head())

    traffic_junction1 = new_traffic.loc[new_traffic["Junction"] == 1]
    traffic_junction2 = new_traffic.loc[new_traffic["Junction"] == 2]
    traffic_junction3 = new_traffic.loc[new_traffic["Junction"] == 3]
    traffic_junction4 = new_traffic.loc[new_traffic["Junction"] == 4]

    traffic_merge = traffic_junction1.merge(traffic_junction2,how='left', left_on='DateTime', right_on='DateTime')
    traffic_merge.rename(columns={'Vehicles_x': 'Junction_1', 'Vehicles_y': 'Junction_2'}, inplace=True)
    traffic_merge = traffic_merge.merge(traffic_junction3,how='left', left_on='DateTime', right_on='DateTime')
    traffic_merge = traffic_merge.merge(traffic_junction4,how='left', left_on='DateTime', right_on='DateTime')
    traffic_merge.rename(columns={'Vehicles_x': 'Junction_3', 'Vehicles_y': 'Junction_4'}, inplace=True)

    traffic_merge = traffic_merge.drop(columns = ['Junction_x','Junction_y'])
    # st.markdown('Dataset with each of the 4 junctions seperated which shows the number of vehicles at a specific datetime:')
    # st.write(traffic_merge.head())

    traffic_merge['Junction_4'] = traffic_merge['Junction_4'].fillna(0)
    traffic_merge['Junction_1'] = traffic_merge['Junction_1'].astype(float)
    traffic_merge['Junction_2'] = traffic_merge['Junction_2'].astype(float)
    traffic_merge['Junction_3'] = traffic_merge['Junction_3'].astype(float)
    st.markdown('Dataset with each of the 4 junctions seperated which shows the number of vehicles at a specific datetime:')
    st.write(traffic_merge.head())

    st.markdown('Timeseries of no. of vehicles at each junction against datetime:')
    fig = traffic_merge.plot(figsize = (20,10))

    split_data = round(len(traffic_merge)*split_ratio)
    split_data

    train_data = traffic_merge[:-split_data]
    test_data = traffic_merge[-split_data:]
    train_time = traffic_merge.index[:-split_data ]
    test_time = traffic_merge.index[-split_data :]
    
    st.markdown("train_data_shape")
    train_data.shape
    st.markdown("test_data_shape")
    test_data.shape

    scaler = MinMaxScaler().fit(train_data.values.reshape(-1,1))
    train_data_normalized  = scaler.transform(train_data.values.reshape(-1, 1))
    test_data_normalized = scaler.transform(test_data.values.reshape(-1, 1))
    st.markdown("train_data_normalized_demand"+str(train_data_normalized.shape))
    st.markdown("test_data_normalized_demand"+str(test_data_normalized.shape))

    # train_data_normalized = train_data_normalized.reshape(train_data.shape[0],train_data.shape[1])
    # st.markdown("test_data_normalized"+str(train_data_normalized.shape))
    
    # test_data_normalized = test_data_normalized.reshape(test_data.shape[0],test_data.shape[1])
    # st.markdown("test_data_normalized"+str(test_data_normalized.shape))

    with echo:
        # st.write('Code will be executed and printed')
        train_data_normalized = train_data_normalized.reshape(train_data.shape[0],train_data.shape[1])
        print("test_data_normalized"+str(train_data_normalized.shape))
        
        test_data_normalized = test_data_normalized.reshape(test_data.shape[0],test_data.shape[1])
        print("test_data_normalized"+str(test_data_normalized.shape))
    st.markdown("test_data_normalized"+str(train_data_normalized.shape))
    st.markdown("test_data_normalized"+str(test_data_normalized.shape))

    def multivariate_univariate_multi_step(sequence,window_size,n_multistep):
        x, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + window_size
            out_ix = end_ix + n_multistep -1
            # check if we are beyond the sequence
            if out_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix,:], sequence[end_ix-1:out_ix,-1]
            x.append(seq_x)
            y.append(seq_y)
            return np.array(x), np.array(y)

    trainX ,trainY=  multivariate_univariate_multi_step(train_data_normalized,window_size,n_step)
    testX , testY = multivariate_univariate_multi_step(test_data_normalized,window_size,n_step)
    trainY = trainY.reshape(trainY.shape[0],n_step,1)
    testY= testY.reshape(testY.shape[0],n_step,1)

    trainX_shape = trainX.shape
    st.write(f"trainX_demand shape:{trainX.shape} trainY_demand shape:{trainY.shape}\n")
    st.write(f"testX_demand shape:{testX.shape} testY_demand shape:{testY.shape}")
