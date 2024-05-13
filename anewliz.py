import warnings

# verileri düzenlemek için
import numpy as np
import pandas as pd

# verileri görselleştirmek için
import seaborn as sns
import matplotlib.pyplot as plt

#Calssifier'lar verileri sınıflandırmak için
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# verileri ölçeklendirmek ve düzenlemek için
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

# başarı oranı için
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

warnings.simplefilter(action='ignore', category=Warning)


# veri output düzenlemek için ayar
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

############# exploratory data analysis EDA  #################

# csv dosyasını okuduk
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

print(df.head())
# son kısımları göstermek için son 10 kısım default 5
print(df.tail(10))
# satır sütün saysı
print(df.shape)