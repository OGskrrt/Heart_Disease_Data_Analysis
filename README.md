# Exploratory Data Analysis and Feature Engineering

This repository contains a script for performing exploratory data analysis (EDA) and feature engineering on a heart disease dataset. The dataset consists of records from Cleveland, Hungary, Switzerland, and Long Beach V, with 76 attributes. However, most analyses focus on a subset of 14 key attributes. The "target" field indicates the presence of heart disease (0 = no disease, 1 = disease).
## This repository made for **Akbank Data Analysis Bootcamp**

## Key Steps Included in the Script:

### 1. Data Loading and Inspection

- **Load Data**:
  - Reads the dataset from a CSV file (`heart_statlog_cleveland_hungary_final.csv`).

- **Inspect Data**:
  - Displays basic information about the DataFrame, including:
    - Shape: The number of rows and columns.
    - Column names: The names of the columns.
    - First and last few rows of the dataset.
    - Data types of each column.
  - **Example Output**:
    ```plaintext
    ##################### Row and Column Count #####################
    (1190, 12)

    ##################### Column Names #####################
    Index(['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope', 'target'], dtype='object')

    ##################### First Five Rows #####################
       age  sex  chest pain type  resting bp s  cholesterol  fasting blood sugar  resting ecg  max heart rate  exercise angina  oldpeak  ST slope  target
    0   40    1                2           140          289                    0            0             172                0    0.000         1       0
    1   49    0                3           160          180                    0            0             156                0    1.000         2       1
    2   37    1                2           130          283                    0            1              98                0    0.000         1       0
    3   48    0                4           138          214                    0            0             108                1    1.500         2       1
    4   54    1                3           150          195                    0            0             122                0    0.000         1       0
    ```

### 2. Basic Statistical Inspection

- **Statistical Summary**:
  - Provides a statistical summary of numerical features, including:
    - Mean
    - Median
    - Mode
    - Standard deviation
  - **Example Output**:
    ```plaintext
                        count    mean     std    min     25%     50%     75%     max
    age                 1190.000  53.720   9.358 28.000  47.000  54.000  60.000  77.000
    sex                 1190.000   0.764   0.425  0.000   1.000   1.000   1.000   1.000
    chest pain type     1190.000   3.233   0.935  1.000   3.000   4.000   4.000   4.000
    resting bp s        1190.000 132.154  18.369  0.000 120.000 130.000 140.000 200.000
    cholesterol         1190.000 210.364 101.420  0.000 188.000 229.000 269.750 603.000
    fasting blood sugar 1190.000   0.213   0.410  0.000   0.000   0.000   0.000   1.000
    resting ecg         1190.000   0.698   0.870  0.000   0.000   0.000   2.000   2.000
    max heart rate      1190.000 139.733  25.518 60.000 121.000 140.500 160.000 202.000
    exercise angina     1190.000   0.387   0.487  0.000   0.000   0.000   1.000   1.000
    oldpeak             1190.000   0.923   1.086 -2.600   0.000   0.600   1.600   6.200
    ST slope            1190.000   1.624   0.610  0.000   1.000   2.000   2.000   3.000
    target              1190.000   0.529   0.499  0.000   0.000   1.000   1.000   1.000
    ```

- **Missing Data Count**:
  - Prints the count of missing values in each column.
  - **Example Output**:
    ```plaintext
    age                    0
    sex                    0
    chest pain type        0
    resting bp s           0
    cholesterol            0
    fasting blood sugar    0
    resting ecg            0
    max heart rate         0
    exercise angina        0
    oldpeak                0
    ST slope               0
    target                 0
    dtype: int64
    ```

### 3. Data Visualization

- **Histograms**: Plots histograms for numerical columns to visualize data distributions.
- **Scatter Plots**: Plots scatter plots for numerical columns against the DataFrame index to identify patterns.
- **Box Plots**: Visualizes the distribution of numerical features and identifies outliers.
- **Correlation Matrix**: Visualizes the correlation between features using a heatmap.

### 4. Data Preprocessing

- **Handle Missing Values**: Checks and prints the count of missing values in each column.
- **Remove Duplicates**: Checks for and removes duplicated rows.
- **Outliers Detection and Handling**: Detects outliers using interquartile range (IQR) thresholds and replaces outliers with threshold values.

### 5. Feature Engineering

- **Normalization**: Applies `RobustScaler` to normalize numerical features, making them more robust to outliers.
- **Feature Selection**: Selects features with high correlation to the target variable for better model performance.
