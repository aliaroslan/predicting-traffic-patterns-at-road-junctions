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
