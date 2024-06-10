

### 1. Import Necessary Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Read Dataset

```python
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Life Expectancy Data.csv")
```

### 3. Sanity Check of Data

```python
# Display the first few rows of the dataframe
df.head()

# Display the last few rows of the dataframe
df.tail()

# Get the shape of the dataframe
df.shape

# Get info on the dataframe
df.info()

# Check for missing values
df.isnull().sum() / df.shape[0] * 100

# Check for duplicate rows
df.duplicated().sum()
```

### 4. Exploratory Data Analysis (EDA)

#### Summary Statistics

```python
df.describe()
```

#### Distribution Plots

```python
for column in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=column, kde=True)
    plt.show()
```

#### Box Plots

```python
for column in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df, x=column)
    plt.show()
```

### 5. Data Cleaning

#### Handling Missing Values

```python
# Fill missing values with median (for numerical columns)
for column in df.select_dtypes(include="number").columns:
    df[column].fillna(df[column].median(), inplace=True)

# Fill missing values with mode (for categorical columns)
for column in df.select_dtypes(include="object").columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
```

#### Verifying Missing Values

```python
df.isnull().sum() / df.shape[0] * 100
```

### 6. Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[df.select_dtypes(include="number").columns] = scaler.fit_transform(df.select_dtypes(include="number"))
```

### 7. Encoding of Data

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Country'] = encoder.fit_transform(df['Country'])
df['Status'] = encoder.fit_transform(df['Status'])
```

### 8. Selecting the Model

#### Splitting the Data

```python
from sklearn.model_selection import train_test_split

X = df.drop('Life expectancy', axis=1)
y = df['Life expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Training a Model (Example: Linear Regression)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

### 9. Deployment and Monitoring

For deployment, we could use Flask or FastAPI to create a web service. For monitoring, tools like Prometheus and Grafana can be used to track the performance and health of the deployed model. Here's an example using Flask:

#### Deployment with Flask

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Save the model
joblib.dump(model, 'model.pkl')

# Load the model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([list(data.values())])
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
```

#### Monitoring with Prometheus and Grafana

1. **Setup Prometheus**: Configure Prometheus to scrape metrics from the Flask app.
2. **Integrate with Flask**: Use `prometheus_flask_exporter` to export metrics from the Flask app.
3. **Visualize with Grafana**: Create dashboards in Grafana to monitor these metrics.

By following these steps, you can build, deploy, and monitor a machine learning model based on the provided life expectancy data.
