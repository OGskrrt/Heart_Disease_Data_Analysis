import warnings
import os

# verileri düzenlemek için
import numpy as np
import pandas as pd

# verileri görselleştirmek için
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


warnings.simplefilter(action='ignore', category=Warning)


# veri output düzenlemek için ayar
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

############# exploratory data analysis EDA  #################

# csv dosyasını okuduk
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# print(df.head())
# son kısımları göstermek için son 10 kısım default 5
# print(df.tail(10))
# satır sütün saysı
# print(df.shape)

def check_df(dataframe):
    print("##################### Row and Column Count #####################")
    print(dataframe.shape)
    print("\n##################### Column Names #####################")
    print(dataframe.columns)
    print("\n##################### First Five Rows #####################")
    print(dataframe.head())
    print("\n##################### Last Five Rows #####################")
    print(dataframe.tail())
    print("\n##################### DataFrame Information #####################")
    dataframe.info()
    print("\n##################### Data Types #####################")
    print(dataframe.dtypes)


check_df(df)


print("\n##################### Statistical Summary #####################")
print(df.describe())


print("\n##################### Missing Data Count #####################")



def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


cols = [col for col in df.columns]

for col in cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        plot_numerical_col(df, col)


#### data preprocessing section

print(df.isnull().sum())

# Check for duplicated datas
df.duplicated().sum()
df = df.drop_duplicates()
df.duplicated().sum()

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    # ooutlier_thresholds ile çeyreklilk değerler belirlenip alt ve üst limitler belirleniyor,
    # limitlerin arasında olmayan değer var ise aykırı değer oluyor
    # .any kısmı tüm satırlara bakıyor
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# check outlier ile aykırı değer olup olmadığına bak
for col in cols:
   print(col, check_outlier(df, col))

# aykırı değeri üst limit veya alt limit ile değiştir
replace_with_thresholds(df, "resting bp s")

for col in cols:
    print(col, check_outlier(df, col))

