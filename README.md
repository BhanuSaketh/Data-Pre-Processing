# Life Expectancy Prediction Project

This project involves building a machine learning model to predict life expectancy based on various health and socioeconomic factors. The steps include data loading, exploratory data analysis, data cleaning, normalization, encoding, model selection, and deployment.

## Project Steps

1. **Import necessary libraries**: Essential libraries for data manipulation, visualization, and modeling.
2. **Read Dataset**: Load the dataset into a pandas DataFrame.
3. **Sanity check of data**: Initial inspection of data to understand its structure and identify any obvious issues.
4. **Exploratory Data Analysis (EDA)**: Analyze and visualize the data to uncover patterns, correlations, and insights.
5. **Data Cleaning**: Handle missing values, outliers, and duplicate entries.
6. **Normalization**: Standardize the data to ensure all features contribute equally to the model.
7. **Encoding of Data**: Convert categorical data into numerical format for machine learning algorithms.
8. **Selecting the model**: Train and evaluate different models to select the best one.
9. **Deployment and Monitoring**: Deploy the model and set up monitoring to ensure it performs well in production.

## Installation

To run this project, you'll need to install the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used in this project is "Life Expectancy Data.csv", which contains various health and socioeconomic indicators for different countries and years.

## Code Overview

Here's a brief overview of the key steps in the code:

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### 2. Load Dataset

```python
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Life Expectancy Data.csv")
```

### 3. Sanity Check

```python
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
```

### 4. Exploratory Data Analysis

```python
df.describe()
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i, kde=True)
    plt.show()
    sns.boxplot(data=df, x=i)
    plt.show()
    sns.scatterplot(data=df, x=i, y="Life expectancy ")
    plt.show()
s = df.select_dtypes(include="number").corr()
plt.figure(figsize=(15, 15))
sns.heatmap(s, annot=True)
```

### 5. Data Cleaning

```python
df.isnull().sum()/df.shape[0]*100
df.drop_duplicates(inplace=True)
im = KNNImputer()
for i in df.select_dtypes(include="number").columns:
    df[i] = im.fit_transform(df[[i]])

def wisker(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lw = q1 - (1.5 * iqr)
    uw = q3 + (1.5 * iqr)
    return lw, uw

for i in df.select_dtypes(include="number").columns:
    lw, uw = wisker(df[i])
    df[i] = np.where(df[i] < lw, lw, df[i])
    df[i] = np.where(df[i] > uw, uw, df[i])
```

### 6. Normalization

```python
sc = StandardScaler()
df['percentage expenditure'] = sc.fit_transform(df['percentage expenditure'].values.reshape(-1, 1))
df['Measles '] = sc.fit_transform(df['Measles '].values.reshape(-1, 1))
```

### 7. Encoding of Data

```python
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])
df['Country'] = le.fit_transform(df['Country'])
```

### 8. Model Selection

#### Linear Regression

```python
X = df.drop("Life expectancy ", axis=1)
y = df["Life expectancy "]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Linear Regression")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
```

#### Decision Tree

```python
dt = DecisionTreeRegressor()
dt.fit(df.drop("Life expectancy ", axis=1), df["Life expectancy "])
predictions = dt.predict(df.drop("Life expectancy ", axis=1))
mae = mean_absolute_error(df["Life expectancy "], predictions)
mse = mean_squared_error(df["Life expectancy "], predictions)
r2 = r2_score(df["Life expectancy "], predictions)
print("Decision Tree")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
```

## Results

- **Linear Regression**
  - Mean Absolute Error: 2.63
  - Mean Squared Error: 12.05
  - R-squared: 0.86

- **Decision Tree**
  - Mean Absolute Error: 0.0
  - Mean Squared Error: 0.0
  - R-squared: 1.0

## Deployment and Monitoring

For deployment, you can use services like AWS SageMaker, Google AI Platform, or Azure ML. Monitoring can be set up using tools like Prometheus and Grafana to ensure the model performs well over time.

## Conclusion

This project demonstrates a comprehensive approach to predicting life expectancy using machine learning. The steps include data preparation, model selection, and evaluation, ensuring a robust and accurate prediction model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
