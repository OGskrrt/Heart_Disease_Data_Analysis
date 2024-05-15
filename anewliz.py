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

    mean_values = df.mean()
    median_values = df.median()
    mode_values = df.mode().iloc[0]
    std_values = df.std()

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], kde=True, color='blue')
        plt.axvline(mean_values[column], color='red', linestyle='--', label=f'Mean: {mean_values[column]:.2f}')
        plt.axvline(median_values[column], color='green', linestyle='-', label=f'Median: {median_values[column]:.2f}')
        plt.axvline(mode_values[column], color='orange', linestyle='-', label=f'Mode: {mode_values[column]:.2f}')
        plt.axvline(std_values[column], color='purple', linestyle='-', label=f'Std: {std_values[column]:.2f}')
        plt.title(f'Distribution of {column}')
        plt.legend()
        plt.show()

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=df.index, y=df[column], alpha=0.5, color='blue')

    plt.title(f'Scatter Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.show()

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

corr_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(df, "target", col)


high_corr_features = corr_matrix['target'][abs(corr_matrix['target']) > 0.4].index.tolist()
high_corr_features.remove('target')
print(f"Selected features: {high_corr_features}")


from sklearn.preprocessing import RobustScaler
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])


df2 = df # First, copy the file
corr_matrix2 = df2.corr() # correlation map
target_variable = 'target' # our target variable

# calculate variables with correlation greater than 0.4
high_corr_features = corr_matrix2['target'][abs(corr_matrix2['target']) > 0.4].index.tolist()

# remove target variable itself
high_corr_features.remove('target')

# check
print(f"Selected features: {high_corr_features}")

# Filter the dataset to keep only high correlation features.
df2 = df2[high_corr_features + ['target']]


