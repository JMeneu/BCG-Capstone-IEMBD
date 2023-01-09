SEED = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 1000
SOURCE_PATH = '/Users/jorgemeneumoreno/Documents/GitHub/[ML] BCG Churn Prediction/BCG-Churn-Prediction/data/data.parquet'
TARGET_PATH = '/Users/jorgemeneumoreno/Documents/GitHub/[ML] BCG Churn Prediction/BCG-Churn-Prediction/results/results.csv'

# General
import os
from mlTools import dataLoader, dataSplitter, dataExplorer, dataProcessor
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import plotly.express as px
import phik
import seaborn as sns
from pandas.plotting import scatter_matrix

# RFM
from rfm import RFM

# CLV
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix 
from lifetimes.plotting import plot_period_transactions
from lifetimes.plotting import plot_cumulative_transactions 
from lifetimes.plotting import plot_incremental_transactions 

# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Feature Extraction
from sklearn.decomposition import PCA

# Anomaly 
from sklearn.ensemble import IsolationForest

# Preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Training & Validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Deployment
import pickle



def data_optimize(df):
    df.product_id= df.product_id.astype(np.int32)
    df.client_id= df.client_id.astype(np.int32)
    df.quantity= df.quantity.astype(np.int32)
    df.branch_id= df.branch_id.astype(np.int16)
    df.sales_net=df.sales_net.astype(np.float32)
    df['date_order'] =  pd.to_datetime(df['date_order'], format='%Y-%m-%d')
    df['date_invoice'] =  pd.to_datetime(df['date_invoice'], format='%Y-%m-%d')
    df.order_channel= df.order_channel.astype('category')
    return df

def drop_values(df, feature):
    df[feature] = df[feature].dropna()
    return df

def drop_erroneous(df, feature, threshold):
    return df[df[feature]>threshold]

def encoder_processor(df, feature):
    df = pd.get_dummies(df, columns = feature)
    return df

def features_keeper(df, features):
    return df[features]

def extract_dates(df):
    df['date_order'] =  pd.to_datetime(df['date_order'], format='%Y-%m-%d')
    df['data_order_year']=df['date_order'].dt.year
    df = encoder_processor(df, ['data_order_year'])
    df['data_order_quarter']=df['date_order'].dt.quarter
    df = encoder_processor(df, ['data_order_quarter'])
    return df

def type_casting(df):
    df.quantity = df.quantity.astype(np.float32)
    df.sales_net = df.sales_net.astype(np.float32)
    df.sales_net=df.sales_net.astype(np.float32)
    df.data_order_year_2017=df.data_order_year_2017.astype(np.uint8)
    df.data_order_year_2018=df.data_order_year_2018.astype(np.uint8)
    df.data_order_year_2019=df.data_order_year_2019.astype(np.uint8)
    df.data_order_quarter_1=df.data_order_quarter_1.astype(np.uint8)
    df.data_order_quarter_2=df.data_order_quarter_2.astype(np.uint8)
    df.data_order_quarter_3=df.data_order_quarter_3.astype(np.uint8)
    df.data_order_quarter_4=df.data_order_quarter_4.astype(np.uint8)
    return df

def numerical_scaler(df, numerical):
    '''
    Scales the numerical features, with an StandardScaler()/MinMaxScaler()
    '''
    scaler = MinMaxScaler()
    df[numerical] = scaler.fit_transform(df[numerical])
    return df

def trend_processor(df, periods=1):
    df = df.sort_values(['client_id', 'date_order'])
    df['prev_order'] = df['date_order'].shift(periods)
    df['delay'] = df['date_order'] - df['prev_order']
    df['avg_delay'] = df.groupby('client_id')['delay'].transform('mean')
    df['trend_pred'] = df['delay'].gt(df['avg_delay']).astype(int)
    df['trend_pred'] = df.groupby('client_id')['trend_pred'].transform(lambda x: 1 if (x == 1).any() else 0)
    num_orders = df.groupby('client_id').size()
    last_order_date = df.groupby('client_id')['date_order'].last()
    if ((num_orders < periods) & last_order_date.dt.year.eq(2019) & last_order_date.dt.quarter.eq(4)).any():
        df['trend_pred']=0
    df = df.drop(['prev_order', 'delay', 'avg_delay'], axis = 1)
    df['trend_pred'] = df['trend_pred'].astype(np.uint8)
    return df

def rfm_processor(df):
    df.quantity = df.quantity.astype(np.int64)
    r = RFM(df, customer_id='client_id', transaction_date='date_order', amount='quantity') 
    r.rfm_table["client_id"] = r.rfm_table["client_id"].astype('int64')
    df = df.merge(r.rfm_table, on='client_id', how='inner')
    enc = OrdinalEncoder()
    df[['segment']] = enc.fit_transform(df[['segment']])
    df = df.drop(['r', 'f', 'm', 'rfm_score'], axis = 1)
    df = numerical_scaler(df, ['recency', 'frequency', 'monetary_value'])
    df['rfm_pred'] = df['segment'].apply(lambda x: 0 if x < 6 else 1)
    df = df.drop('segment', axis = 1)
    return df

def cluster_processor(df, k):
    kmeans = KMeans(n_clusters = k, random_state = SEED, max_iter=1000).fit(df)
    df['kmeans_pred'] = kmeans.predict(df)
    return df

def anomaly_processor(df):
    model = IsolationForest(random_state=SEED)
    model.fit(df)
    pred = model.predict(df)
    df['anomaly_pred'] = pred
    df['anomaly_pred'] = df['anomaly_pred'].replace(-1, 1).replace(1, 0)
    return df

def churn_imputer(df):
    df['majority'] = df[['trend_pred', 'rfm_pred', 'kmeans_pred', 'anomaly_pred']].sum(axis=1)
    df['Churn'] = df['majority'].apply(lambda x: 1 if x >= 2 else 0)
    df = df.drop(['trend_pred', 'rfm_pred', 'kmeans_pred', 'anomaly_pred', 'majority'], axis = 1)
    return df

def preprocessor(df):
    df = data_optimize(df)
    df = drop_values(df, 'date_invoice')
    df = drop_erroneous(df, 'sales_net', 0)
    df = encoder_processor(df, ['order_channel'])
    df = features_keeper(df, ['client_id','date_order','quantity','sales_net'])
    df = extract_dates(df)
    df = type_casting(df)
    df = numerical_scaler(df, ['quantity', 'sales_net']) 
    df = type_casting(df)
    df = trend_processor(df, 1)
    df = type_casting(df)
    df = rfm_processor(df)
    df = df.drop(['date_order'], axis = 1)
    df['sales_net'] = pd.to_numeric(df['sales_net'], errors='coerce')
    df = type_casting(df)
    df = df.dropna()
    df = cluster_processor(df, 2)
    df = anomaly_processor(df)
    df['Churn'] = 0
    df = churn_imputer(df)
    return df

def train_processor(df, is_Train):
    X = train.drop(['client_id', 'Churn'], axis = 1)
    y = train['Churn']
    model = RandomForestClassifier(max_depth=50, 
                                max_features='sqrt',
                                min_samples_leaf=40,
                                min_samples_split=30,
                                n_estimators=1000, 
                                random_state = SEED)
    model.fit(X, y)
    pred = model.predict(X)
    if is_Train:
        print("[4/6] Train Processor: Done! [ACC: ", accuracy_score(y, pred), " ]")
        return model
    else:
        print("[5/6] Test Processor: Done! [ACC: ", accuracy_score(y, pred), " ]")
        return pred

def export_processor(pred, model, path):
    submission = pd.DataFrame(data = pred, columns = ['Churn'])
    submission['client_id']=test['client_id']
    submission.set_index('client_id', inplace = True)
    submission.to_csv(path)
    pickle.dump(model, open('model.pkl', 'wb'))

loaderObj = dataLoader()
data = loaderObj.batch_loader(SOURCE_PATH,False)
print("[1/6] Data Loader: Done!")

splitterObj = dataSplitter(data)
train,test = splitterObj.train_splitter("sales_net", TEST_SIZE, SEED, False)
print("[2/6] Data Splitter: Done!")

train = train.sample(SAMPLE_SIZE)
test = test.sample(int(SAMPLE_SIZE*TEST_SIZE))


train = preprocessor(train)
test = preprocessor(test)
print("[3/6] Data Preprocessor: Done!")

model = train_processor(train, True)
pred = train_processor(test, False)


export_processor(model, pred, TARGET_PATH)
print("[6/6] Export Processor: Done!")